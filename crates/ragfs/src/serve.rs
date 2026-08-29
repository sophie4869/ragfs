//! Local HTTP query server (`ragfs serve`).
//!
//! Keeps the embedding model and index loaded so an interactive client (e.g.
//! the Obsidian plugin) gets fast, repeated queries instead of paying the
//! per-invocation model reload of `ragfs query`.

use anyhow::Context;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use ragfs_core::{SearchResult, VectorStore};
use ragfs_embed::EmbedderPool;
use ragfs_query::QueryExecutor;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

/// Max results a single request may ask for.
const MAX_LIMIT: usize = 100;

/// Length (chars) of the content snippet returned per result.
const SNIPPET_LEN: usize = 240;

/// Shared server state: the model and index, loaded once.
struct AppState {
    store: Arc<dyn VectorStore>,
    embedder: Arc<EmbedderPool>,
    default_limit: usize,
    model: String,
    index_path: String,
}

/// Run the local query server until the process is stopped.
///
/// `store` and `embedder` are already-initialized shared components, so the
/// embedding model is loaded exactly once for the lifetime of the server.
pub async fn run(
    store: Arc<dyn VectorStore>,
    embedder: Arc<EmbedderPool>,
    model: String,
    index_path: String,
    host: &str,
    port: u16,
    default_limit: usize,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        store,
        embedder,
        default_limit,
        model,
        index_path,
    });

    // Localhost-only server; a permissive CORS policy lets the Obsidian plugin
    // (origin app://obsidian.md) and browsers call it.
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/query", get(query_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind {addr}"))?;
    info!("ragfs serve listening on http://{addr}");
    info!("  GET /query?q=<text>&limit=<n>");

    axum::serve(listener, app)
        .await
        .context("HTTP server error")?;
    Ok(())
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
    index_path: String,
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.model.clone(),
        index_path: state.index_path.clone(),
    })
}

async fn query_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let query = params.get("q").map_or("", String::as_str).trim();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing required query parameter `q`".to_string(),
        ));
    }
    let limit = parse_limit(params.get("limit").map(String::as_str), state.default_limit);

    let executor = QueryExecutor::new(
        state.store.clone(),
        state.embedder.document_embedder(),
        limit,
        false, // vector-only; hybrid is opt-in and still being fixed
    );

    let results = executor
        .execute(query)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(to_response(query, &results, SNIPPET_LEN)))
}

/// One search hit in the HTTP response.
#[derive(Serialize)]
pub struct ResultItem {
    pub file: String,
    pub score: f32,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines: Option<String>,
}

/// Response body for `GET /query`.
#[derive(Serialize)]
pub struct QueryResponse {
    pub query: String,
    pub results: Vec<ResultItem>,
}

/// Resolve the effective result limit from a raw query-string value.
///
/// Missing or unparseable values fall back to `default`; the result is clamped
/// to `[1, MAX_LIMIT]` so a client cannot ask for zero or an unbounded number.
pub fn parse_limit(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(1, MAX_LIMIT)
}

/// Build the HTTP response body from raw search results, truncating snippets.
pub fn to_response(query: &str, results: &[SearchResult], snippet_len: usize) -> QueryResponse {
    QueryResponse {
        query: query.to_string(),
        results: results
            .iter()
            .map(|r| ResultItem {
                file: r.file_path.to_string_lossy().to_string(),
                score: r.score,
                content: truncate(&r.content, snippet_len),
                lines: r
                    .line_range
                    .as_ref()
                    .map(|l| format!("{}:{}", l.start, l.end)),
            })
            .collect(),
    }
}

/// Collapse whitespace-ish characters and cap length.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace(['\n', '\r'], " ");
    if s.chars().count() <= max_len {
        s
    } else {
        let cut: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{cut}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn result(path: &str, score: f32, content: &str, lines: Option<(u32, u32)>) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::nil(),
            file_path: PathBuf::from(path),
            content: content.to_string(),
            score,
            byte_range: 0..0,
            line_range: lines.map(|(a, b)| a..b),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_parse_limit_default_and_bounds() {
        assert_eq!(parse_limit(None, 10), 10);
        assert_eq!(parse_limit(Some("5"), 10), 5);
        assert_eq!(parse_limit(Some("abc"), 10), 10); // unparseable -> default
        assert_eq!(parse_limit(Some("0"), 10), 1); // clamped up to 1
        assert_eq!(parse_limit(Some("9999"), 10), MAX_LIMIT); // clamped down
    }

    #[test]
    fn test_to_response_maps_fields() {
        let results = vec![
            result("/vault/a.md", 0.82, "hello world", Some((3, 9))),
            result("/vault/b.md", 0.50, "no lines here", None),
        ];
        let resp = to_response("q", &results, 100);

        assert_eq!(resp.query, "q");
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].file, "/vault/a.md");
        assert_eq!(resp.results[0].lines.as_deref(), Some("3:9"));
        assert_eq!(resp.results[1].lines, None);
    }

    #[test]
    fn test_to_response_truncates_snippet() {
        let results = vec![result("/v/x.md", 1.0, "abcdefghij", None)];
        let resp = to_response("q", &results, 6);
        assert_eq!(resp.results[0].content, "abc..."); // 6-3 kept + ellipsis
    }
}
