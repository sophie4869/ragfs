//! Local HTTP query server (`ragfs serve`).
//!
//! Keeps the embedding model and index loaded so an interactive client (e.g.
//! the Obsidian plugin) gets fast, repeated queries instead of paying the
//! per-invocation model reload of `ragfs query`.

use anyhow::Context;
use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use ragfs_core::{Chunk, FileRecord, SearchResult, StoreStats, VectorStore};
use ragfs_embed::EmbedderPool;
use ragfs_query::QueryExecutor;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

/// Max results a single request may ask for.
const MAX_LIMIT: usize = 100;

/// Max vector candidates fetched before HTTP-layer reranking.
const MAX_CANDIDATE_LIMIT: usize = 100;

/// Length (chars) of the content snippet returned per result.
const SNIPPET_LEN: usize = 240;

const INDEX_HTML: &str = include_str!("web/index.html");
const APP_CSS: &str = include_str!("web/app.css");
const APP_JS: &str = include_str!("web/app.js");
const MATCH_REASON_KEY: &str = "match_reason";

/// Shared server state: the model and index, loaded once.
struct AppState {
    store: Arc<dyn VectorStore>,
    embedder: Arc<EmbedderPool>,
    default_limit: usize,
    model: String,
    index_path: String,
    root_path: PathBuf,
    token: Option<String>,
    query_lock: Mutex<()>,
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
    token: Option<String>,
) -> anyhow::Result<()> {
    let root_path = PathBuf::from(&index_path)
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize index path {index_path}"))?;
    let state = Arc::new(AppState {
        store,
        embedder,
        default_limit,
        model,
        index_path,
        root_path,
        token,
        query_lock: Mutex::new(()),
    });

    // Localhost-only server; a permissive CORS policy lets the Obsidian plugin
    // (origin app://obsidian.md) and browsers call it.
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any);

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/app.css", get(css_handler))
        .route("/app.js", get(js_handler))
        .route("/health", get(health_handler))
        .route("/query", get(query_handler))
        .route("/api/search", get(query_handler))
        .route("/api/status", get(status_handler))
        .route("/api/files/{*path}", get(file_handler))
        .route("/raw/{*path}", get(raw_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind {addr}"))?;
    info!("ragfs serve listening on http://{addr}");
    info!("  GET /query?q=<text>&limit=<n>");
    info!("  GET /api/search?q=<text>&limit=<n>");
    info!("  GET /api/files/<relative-path>");
    info!("  GET /raw/<relative-path>");

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

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn css_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static("text/css"))],
        APP_CSS,
    )
}

async fn js_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/javascript"),
        )],
        APP_JS,
    )
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
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    require_auth(&state, &headers)?;

    let query = params.get("q").map_or("", String::as_str).trim();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing required query parameter `q`".to_string(),
        ));
    }
    let limit = parse_limit(params.get("limit").map(String::as_str), state.default_limit);
    let candidate_limit = candidate_limit(limit);
    let profile = QueryProfile::new(query);
    let search_queries = semantic_queries(query, &profile);

    let vector_results = {
        // Candle's Metal backend can abort on concurrent command encoders. Keep
        // embedding-backed searches single-flight; the rest of the HTTP server
        // remains concurrent.
        let _guard = state.query_lock.lock().await;
        execute_semantic_queries(&state, &search_queries, candidate_limit).await?
    };
    let lexical_results =
        lexical_candidates(&profile, &state.store, &state.root_path, candidate_limit)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let results = rerank_results(
        &profile,
        vector_results,
        lexical_results,
        &state.root_path,
        limit,
    );

    Ok(Json(to_response_with_root(
        query,
        &results,
        SNIPPET_LEN,
        Some(&state.root_path),
    )))
}

/// One search hit in the HTTP response.
#[derive(Serialize)]
pub struct ResultItem {
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_path: Option<String>,
    pub title: String,
    pub kind: String,
    pub score: f32,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines: Option<String>,
}

/// Response body for `GET /query`.
#[derive(Serialize)]
pub struct QueryResponse {
    pub query: String,
    pub results: Vec<ResultItem>,
}

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
    model: String,
    index_path: String,
    total_files: u64,
    total_chunks: u64,
    index_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_updated: Option<String>,
}

#[derive(Serialize)]
struct FileResponse {
    file: String,
    relative_path: String,
    title: String,
    mime_type: String,
    size_bytes: u64,
    modified_at: String,
    indexed_at: Option<String>,
    chunks: Vec<FileChunkResponse>,
    text: Option<String>,
    raw_url: String,
}

#[derive(Serialize)]
struct FileChunkResponse {
    content: String,
    lines: Option<String>,
}

async fn status_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    require_auth(&state, &headers)?;

    let stats = state
        .store
        .stats()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(status_response(&state, stats)))
}

async fn file_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(path): AxumPath<String>,
) -> Result<Json<FileResponse>, (StatusCode, String)> {
    require_auth(&state, &headers)?;

    let absolute = canonical_file_path(&state.root_path, &path).await?;
    let record = state
        .store
        .get_file(&absolute)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let Some(record) = record else {
        return Err((StatusCode::NOT_FOUND, "file is not indexed".to_string()));
    };
    let chunks = state
        .store
        .get_chunks_for_file(&absolute)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(file_response(&state.root_path, &record, chunks)))
}

async fn raw_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(path): AxumPath<String>,
) -> Result<Response, (StatusCode, String)> {
    require_auth(&state, &headers)?;

    let absolute = canonical_file_path(&state.root_path, &path).await?;
    let record = state
        .store
        .get_file(&absolute)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let Some(record) = record else {
        return Err((StatusCode::NOT_FOUND, "file is not indexed".to_string()));
    };

    let bytes = tokio::fs::read(&record.path)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, format!("failed to read file: {e}")))?;

    let content_type = if record.mime_type.trim().is_empty() {
        mime_guess::from_path(&record.path)
            .first_or_octet_stream()
            .to_string()
    } else {
        record.mime_type
    };
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
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

fn candidate_limit(limit: usize) -> usize {
    (limit.saturating_mul(5)).clamp(limit, MAX_CANDIDATE_LIMIT)
}

/// Build the HTTP response body from raw search results, truncating snippets.
#[cfg(test)]
pub fn to_response(query: &str, results: &[SearchResult], snippet_len: usize) -> QueryResponse {
    to_response_with_root(query, results, snippet_len, None)
}

/// Build the HTTP response body from raw search results, truncating snippets.
pub fn to_response_with_root(
    query: &str,
    results: &[SearchResult],
    snippet_len: usize,
    root: Option<&Path>,
) -> QueryResponse {
    QueryResponse {
        query: query.to_string(),
        results: results
            .iter()
            .map(|r| result_item(r, snippet_len, root))
            .collect(),
    }
}

fn result_item(r: &SearchResult, snippet_len: usize, root: Option<&Path>) -> ResultItem {
    let relative_path = root.and_then(|root| relative_path(&r.file_path, root));
    ResultItem {
        file: r.file_path.to_string_lossy().to_string(),
        relative_path,
        title: title_for_path(&r.file_path),
        kind: kind_for_path(&r.file_path),
        score: r.score,
        content: truncate(&r.content, snippet_len),
        reason: r.metadata.get("match_reason").cloned(),
        lines: r
            .line_range
            .as_ref()
            .map(|l| format!("{}:{}", l.start, l.end)),
    }
}

async fn execute_semantic_queries(
    state: &AppState,
    queries: &[String],
    limit: usize,
) -> Result<Vec<SearchResult>, (StatusCode, String)> {
    let mut results = Vec::new();
    for (index, query) in queries.iter().enumerate() {
        let executor = QueryExecutor::new(
            state.store.clone(),
            state.embedder.document_embedder(),
            limit,
            false, // vector-only; hybrid is opt-in and still being fixed
        );
        let reason = semantic_reason(index, query);
        let weight = if index == 0 { 1.0 } else { 0.96 };

        let mut query_results = executor
            .execute(query)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        for result in &mut query_results {
            result.score *= weight;
            result
                .metadata
                .insert(MATCH_REASON_KEY.to_string(), reason.clone());
        }
        results.extend(query_results);
    }
    Ok(results)
}

fn semantic_reason(index: usize, query: &str) -> String {
    if index == 0 {
        "semantic".to_string()
    } else if query.split_whitespace().count() > 4 {
        "semantic: keywords".to_string()
    } else {
        format!("semantic: {}", truncate(query, 64))
    }
}

async fn lexical_candidates(
    profile: &QueryProfile,
    store: &Arc<dyn VectorStore>,
    root: &Path,
    limit: usize,
) -> Result<Vec<SearchResult>, ragfs_core::StoreError> {
    if profile.terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored = store
        .get_all_chunks()
        .await?
        .into_iter()
        .filter_map(|chunk| {
            let signal = lexical_signal(profile, &chunk.file_path, &chunk.content, root);
            (signal.score > 0.0).then(|| {
                (
                    signal.score,
                    search_result_from_chunk(chunk, signal.score, signal.reason()),
                )
            })
        })
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(scored
        .into_iter()
        .take(limit)
        .map(|(_, result)| result)
        .collect())
}

fn search_result_from_chunk(chunk: Chunk, score: f32, reason: String) -> SearchResult {
    let mut metadata = chunk.metadata.extra;
    metadata.insert(MATCH_REASON_KEY.to_string(), reason);

    SearchResult {
        chunk_id: chunk.id,
        file_path: chunk.file_path,
        content: chunk.content,
        score,
        byte_range: chunk.byte_range,
        line_range: chunk.line_range,
        metadata,
    }
}

fn rerank_results(
    profile: &QueryProfile,
    vector_results: Vec<SearchResult>,
    lexical_results: Vec<SearchResult>,
    root: &Path,
    limit: usize,
) -> Vec<SearchResult> {
    let mut by_file = HashMap::<PathBuf, SearchResult>::new();

    for mut result in vector_results {
        let signal = lexical_signal(profile, &result.file_path, &result.content, root);
        result.score += signal.score.min(0.34);
        result.metadata.insert(
            MATCH_REASON_KEY.to_string(),
            combined_reason(&result, &signal),
        );
        insert_best_by_file(&mut by_file, result);
    }

    for mut result in lexical_results {
        // Lexical-only results need a floor high enough to beat weak semantic
        // neighbors, while still being driven by generic path/title/content
        // overlap rather than hard-coded domain aliases.
        let signal = lexical_signal(profile, &result.file_path, &result.content, root);
        let lexical = result.score.max(signal.score);
        result.score = 0.55 + lexical.min(0.42);
        if !result.metadata.contains_key(MATCH_REASON_KEY) {
            result
                .metadata
                .insert(MATCH_REASON_KEY.to_string(), signal.reason());
        }
        insert_best_by_file(&mut by_file, result);
    }

    let mut results = by_file.into_values().collect::<Vec<_>>();
    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results.truncate(limit);
    results
}

fn insert_best_by_file(results: &mut HashMap<PathBuf, SearchResult>, result: SearchResult) {
    match results.get(&result.file_path) {
        Some(existing) if existing.score >= result.score => {}
        _ => {
            results.insert(result.file_path.clone(), result);
        }
    }
}

#[derive(Debug)]
struct QueryProfile {
    terms: Vec<String>,
    phrases: Vec<String>,
}

impl QueryProfile {
    fn new(query: &str) -> Self {
        let terms = query_terms(query);
        let phrases = query_phrases(&terms);
        Self { terms, phrases }
    }
}

#[derive(Debug, Default)]
struct LexicalSignal {
    score: f32,
    matched_terms: usize,
    reasons: Vec<String>,
    seen_reasons: HashSet<String>,
}

impl LexicalSignal {
    fn add(&mut self, amount: f32, reason: impl Into<String>) {
        self.score += amount;
        let reason = reason.into();
        if self.seen_reasons.insert(reason.clone()) && self.reasons.len() < 4 {
            self.reasons.push(reason);
        }
    }

    fn reason(&self) -> String {
        if self.reasons.is_empty() {
            "lexical".to_string()
        } else {
            self.reasons.join(", ")
        }
    }
}

fn combined_reason(result: &SearchResult, signal: &LexicalSignal) -> String {
    let semantic = result
        .metadata
        .get(MATCH_REASON_KEY)
        .map_or("semantic", String::as_str);
    if signal.score > 0.0 {
        format!("{semantic}, {}", signal.reason())
    } else {
        semantic.to_string()
    }
}

fn lexical_signal(
    profile: &QueryProfile,
    path: &Path,
    content: &str,
    root: &Path,
) -> LexicalSignal {
    let title = normalize_search_text(&title_for_path(path));
    let rel_path = relative_path(path, root).unwrap_or_else(|| path.to_string_lossy().to_string());
    let path_text = normalize_search_text(&rel_path);
    let content_text = normalize_search_text(content);
    let title_tokens = text_tokens(&title);
    let path_tokens = text_tokens(&path_text);
    let content_tokens = text_tokens(&content_text);
    let mut signal = LexicalSignal::default();

    for phrase in profile.phrases.iter().take(8) {
        if title.contains(phrase) {
            signal.add(0.24, format!("title phrase: {phrase}"));
        }
        if path_text.contains(phrase) {
            signal.add(0.16, format!("path phrase: {phrase}"));
        }
        if content_text.contains(phrase) {
            signal.add(0.10, format!("content phrase: {phrase}"));
        }
    }

    for term in &profile.terms {
        let mut term_score: f32 = 0.0;
        if text_matches_term(&title, &title_tokens, term) {
            term_score += 0.18;
            signal.add(0.0, format!("title: {term}"));
        }
        if text_matches_term(&path_text, &path_tokens, term) {
            term_score += 0.13;
            signal.add(0.0, format!("path: {term}"));
        }
        if text_matches_term(&content_text, &content_tokens, term) {
            term_score += 0.05;
            signal.add(0.0, format!("content: {term}"));
        }
        if term_score > 0.0 {
            signal.matched_terms += 1;
            signal.score += term_score.min(0.20);
        }
    }

    if signal.matched_terms >= 2 {
        signal.score += 0.06;
    }
    if signal.matched_terms >= 4 {
        signal.score += 0.04;
    }
    signal.score = signal.score.min(0.70);
    signal
}

fn semantic_queries(query: &str, profile: &QueryProfile) -> Vec<String> {
    let mut queries = Vec::new();
    let mut seen = HashSet::new();
    push_query(&mut queries, &mut seen, query);

    if profile.terms.len() >= 2 {
        push_query(&mut queries, &mut seen, &profile.terms.join(" "));
    }
    for phrase in profile.phrases.iter().take(3) {
        push_query(&mut queries, &mut seen, phrase);
    }

    queries.truncate(5);
    queries
}

fn push_query(queries: &mut Vec<String>, seen: &mut HashSet<String>, query: &str) {
    let query = query.trim();
    if !query.is_empty() && seen.insert(query.to_string()) {
        queries.push(query.to_string());
    }
}

fn query_terms(query: &str) -> Vec<String> {
    let stopwords = stopwords();
    let mut seen = HashSet::new();
    let mut terms = Vec::new();

    for raw in raw_terms(query) {
        for variant in term_variants(&raw) {
            let term = normalize_search_text(&variant);
            if term.chars().count() >= 2
                && !stopwords.contains(term.as_str())
                && seen.insert(term.clone())
            {
                terms.push(term);
            }
        }
    }

    terms
}

fn raw_terms(query: &str) -> Vec<String> {
    let normalized = normalize_search_text(query);
    let mut terms = Vec::new();
    for part in normalized.split_whitespace() {
        if part.chars().any(is_cjk) && part.chars().count() > 4 {
            terms.extend(cjk_ngrams(part, 2));
            terms.extend(cjk_ngrams(part, 3));
        } else {
            terms.push(part.to_string());
        }
    }
    terms
}

fn query_phrases(terms: &[String]) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut seen = HashSet::new();
    for width in 2..=3 {
        for window in terms.windows(width) {
            if window.iter().all(|term| !term.chars().any(is_cjk)) {
                let phrase = window.join(" ");
                if seen.insert(phrase.clone()) {
                    phrases.push(phrase);
                }
            }
        }
    }
    phrases
}

fn term_variants(term: &str) -> Vec<String> {
    let mut variants = vec![term.to_string()];
    if term.chars().any(is_cjk) {
        return variants;
    }

    if let Some(stripped) = term.strip_suffix('s')
        && stripped.len() >= 3
    {
        variants.push(stripped.to_string());
    }
    if let Some(stripped) = term.strip_suffix("ies")
        && stripped.len() >= 2
    {
        variants.push(format!("{stripped}y"));
    } else if let Some(stripped) = term.strip_suffix('y')
        && stripped.len() >= 2
    {
        variants.push(format!("{stripped}ies"));
    }
    if let Some(stripped) = term.strip_suffix("ing")
        && stripped.len() >= 4
    {
        variants.push(stripped.to_string());
    }
    if let Some(stripped) = term.strip_suffix("ed")
        && stripped.len() >= 4
    {
        variants.push(stripped.to_string());
    }
    variants
}

fn cjk_ngrams(input: &str, width: usize) -> Vec<String> {
    let chars = input.chars().collect::<Vec<_>>();
    chars
        .windows(width)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

fn text_tokens(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

fn text_matches_term(text: &str, tokens: &[&str], term: &str) -> bool {
    if term.chars().any(is_cjk) {
        return text.contains(term);
    }
    tokens.iter().any(|token| {
        *token == term
            || (term.len() >= 4 && token.starts_with(term))
            || (token.len() >= 5 && term.starts_with(*token))
    })
}

fn normalize_search_text(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || is_cjk(c) {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn stopwords() -> HashSet<&'static str> {
    [
        "a", "an", "and", "are", "did", "do", "for", "here", "how", "i", "in", "is", "it", "my",
        "note", "notes", "of", "or", "save", "saved", "the", "to", "what", "where", "which",
        "your", "关于", "保存", "哪些",
    ]
    .into_iter()
    .collect()
}

fn status_response(state: &AppState, stats: StoreStats) -> StatusResponse {
    StatusResponse {
        status: "ok",
        model: state.model.clone(),
        index_path: state.index_path.clone(),
        total_files: stats.total_files,
        total_chunks: stats.total_chunks,
        index_size_bytes: stats.index_size_bytes,
        last_updated: stats.last_updated.map(|t| t.to_rfc3339()),
    }
}

fn file_response(
    root: &Path,
    record: &FileRecord,
    mut chunks: Vec<ragfs_core::Chunk>,
) -> FileResponse {
    chunks.sort_by_key(|c| c.chunk_index);
    let text = if is_text_like(&record.mime_type, &record.path) {
        Some(
            chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    } else {
        None
    };
    let relative_path = relative_path(&record.path, root).unwrap_or_else(|| {
        record
            .path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().to_string())
    });

    FileResponse {
        file: record.path.to_string_lossy().to_string(),
        relative_path: relative_path.clone(),
        title: title_for_path(&record.path),
        mime_type: record.mime_type.clone(),
        size_bytes: record.size_bytes,
        modified_at: record.modified_at.to_rfc3339(),
        indexed_at: record.indexed_at.map(|t| t.to_rfc3339()),
        chunks: chunks
            .into_iter()
            .map(|c| FileChunkResponse {
                content: c.content,
                lines: c.line_range.map(|l| format!("{}:{}", l.start, l.end)),
            })
            .collect(),
        text,
        raw_url: format!("/raw/{}", url_path_escape(&relative_path)),
    }
}

fn require_auth(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    if is_authorized(state.token.as_deref(), headers) {
        Ok(())
    } else {
        Err((StatusCode::UNAUTHORIZED, "unauthorized".to_string()))
    }
}

fn is_authorized(token: Option<&str>, headers: &HeaderMap) -> bool {
    let Some(token) = token.filter(|t| !t.is_empty()) else {
        return true;
    };
    let bearer = format!("Bearer {token}");

    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == bearer)
        || headers
            .get("x-ragfs-token")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == token)
}

async fn canonical_file_path(root: &Path, raw: &str) -> Result<PathBuf, (StatusCode, String)> {
    let candidate = resolve_file_path(root, raw)?;
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "file not found".to_string()))?;

    if !canonical.starts_with(root) {
        return Err((StatusCode::FORBIDDEN, "path escapes index root".to_string()));
    }
    if !canonical.is_file() {
        return Err((StatusCode::BAD_REQUEST, "path is not a file".to_string()));
    }
    Ok(canonical)
}

/// Resolve a client path under the indexed root without allowing `..` escapes.
pub fn resolve_file_path(root: &Path, raw: &str) -> Result<PathBuf, (StatusCode, String)> {
    if Path::new(raw).is_absolute() {
        return Err((StatusCode::FORBIDDEN, "invalid file path".to_string()));
    }

    let raw = raw.trim_start_matches('/');
    if raw.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "missing file path".to_string()));
    }

    let mut clean = PathBuf::new();
    for component in Path::new(raw).components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err((StatusCode::FORBIDDEN, "invalid file path".to_string()));
            }
        }
    }

    if clean.as_os_str().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "missing file path".to_string()));
    }

    Ok(root.join(clean))
}

fn relative_path(path: &Path, root: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

fn title_for_path(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .map_or_else(String::new, |n| n.to_string_lossy().to_string())
}

fn kind_for_path(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map_or_else(|| "file".to_string(), str::to_ascii_lowercase)
}

fn is_text_like(mime_type: &str, path: &Path) -> bool {
    mime_type.starts_with("text/")
        || matches!(
            path.extension().and_then(|e| e.to_str()),
            Some(
                "md" | "markdown"
                    | "txt"
                    | "json"
                    | "toml"
                    | "yaml"
                    | "yml"
                    | "rs"
                    | "js"
                    | "ts"
                    | "tsx"
                    | "jsx"
                    | "py"
                    | "css"
                    | "html"
            )
        )
}

fn url_path_escape(path: &str) -> String {
    path.split('/')
        .map(url_segment_escape)
        .collect::<Vec<_>>()
        .join("/")
}

fn url_segment_escape(segment: &str) -> String {
    let mut escaped = String::new();
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                escaped.push(byte as char);
            }
            _ => escaped.push_str(&format!("%{byte:02X}")),
        }
    }
    escaped
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
    fn test_candidate_limit_expands_small_requests() {
        assert_eq!(candidate_limit(3), 15);
        assert_eq!(candidate_limit(25), MAX_CANDIDATE_LIMIT);
        assert_eq!(candidate_limit(MAX_LIMIT), MAX_CANDIDATE_LIMIT);
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
        assert_eq!(resp.results[0].title, "a");
        assert_eq!(resp.results[0].kind, "md");
        assert_eq!(resp.results[0].lines.as_deref(), Some("3:9"));
        assert_eq!(resp.results[1].lines, None);
    }

    #[test]
    fn test_to_response_truncates_snippet() {
        let results = vec![result("/v/x.md", 1.0, "abcdefghij", None)];
        let resp = to_response("q", &results, 6);
        assert_eq!(resp.results[0].content, "abc..."); // 6-3 kept + ellipsis
    }

    #[test]
    fn test_to_response_adds_relative_path_when_root_matches() {
        let results = vec![result("/vault/folder/a.md", 0.82, "hello world", None)];
        let resp = to_response_with_root("q", &results, 100, Some(Path::new("/vault")));
        assert_eq!(
            resp.results[0].relative_path.as_deref(),
            Some("folder/a.md")
        );
    }

    #[test]
    fn test_query_terms_use_generic_variants_not_domain_aliases() {
        let terms = query_terms("where did I save astronomy or night sky photo notes");
        assert!(terms.contains(&"astronomy".to_string()));
        assert!(terms.contains(&"sky".to_string()));
        assert!(terms.contains(&"skies".to_string()));
        assert!(terms.contains(&"photo".to_string()));
        assert!(!terms.contains(&"where".to_string()));
        assert!(!terms.contains(&"stargazing".to_string()));
        assert!(!terms.contains(&"milky".to_string()));
    }

    #[test]
    fn test_query_terms_extract_cjk_ngrams_from_long_queries() {
        let terms = query_terms("关于房东纠纷我保存了哪些证据");
        assert!(terms.contains(&"房东".to_string()));
        assert!(terms.contains(&"纠纷".to_string()));
        assert!(terms.contains(&"证据".to_string()));
        assert!(!terms.contains(&"哪些".to_string()));
    }

    #[test]
    fn test_semantic_queries_add_compact_query_for_long_questions() {
        let profile = QueryProfile::new("where did I save astronomy or night sky photo notes");
        let queries = semantic_queries(
            "where did I save astronomy or night sky photo notes",
            &profile,
        );

        assert_eq!(
            queries[0],
            "where did I save astronomy or night sky photo notes"
        );
        assert!(
            queries
                .iter()
                .any(|query| query.contains("astronomy") && query.contains("photo"))
        );
    }

    #[test]
    fn test_rerank_adds_path_and_content_lexical_candidates() {
        let root = Path::new("/vault");
        let profile = QueryProfile::new("where did I save astronomy or night sky photo notes");
        let vector_results = vec![result(
            "/vault/Incidents/boris.pdf",
            0.64,
            "answer to the code review team and contact people ops",
            None,
        )];
        let lexical_results = vec![result(
            "/vault/Photography/Milky way.md",
            0.0,
            "places to go stargazing in Europe with clear skies",
            None,
        )];

        let reranked = rerank_results(&profile, vector_results, lexical_results, root, 2);

        assert_eq!(
            reranked[0].file_path,
            PathBuf::from("/vault/Photography/Milky way.md")
        );
        assert!(reranked[0].metadata["match_reason"].contains("path: photo"));
        assert!(reranked[0].metadata["match_reason"].contains("content: skies"));
    }

    #[test]
    fn test_rerank_deduplicates_by_file() {
        let root = Path::new("/vault");
        let profile = QueryProfile::new("query");
        let vector_results = vec![
            result("/vault/a.md", 0.70, "first chunk", None),
            result("/vault/a.md", 0.80, "better chunk", None),
            result("/vault/b.md", 0.60, "other", None),
        ];

        let reranked = rerank_results(&profile, vector_results, Vec::new(), root, 10);

        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].file_path, PathBuf::from("/vault/a.md"));
        assert_eq!(reranked[0].content, "better chunk");
    }

    #[test]
    fn test_resolve_file_path_rejects_escapes() {
        let root = Path::new("/vault");
        assert_eq!(
            resolve_file_path(root, "folder/a.md").unwrap(),
            PathBuf::from("/vault/folder/a.md")
        );
        assert!(resolve_file_path(root, "../secret.md").is_err());
        assert!(resolve_file_path(root, "/etc/passwd").is_err());
    }

    #[test]
    fn test_url_path_escape_preserves_path_separators() {
        assert_eq!(
            url_path_escape("Incidents/tripod note.md"),
            "Incidents/tripod%20note.md"
        );
    }

    #[test]
    fn test_is_authorized_accepts_bearer_and_custom_header() {
        let mut headers = HeaderMap::new();
        assert!(is_authorized(None, &headers));
        assert!(!is_authorized(Some("secret"), &headers));

        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert!(is_authorized(Some("secret"), &headers));

        headers.clear();
        headers.insert("x-ragfs-token", HeaderValue::from_static("secret"));
        assert!(is_authorized(Some("secret"), &headers));
    }
}
