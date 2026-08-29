//! # RAGFS CLI
//!
//! Command-line interface for RAGFS (Retrieval-Augmented Generation `FileSystem`).
//!
//! RAGFS enables semantic search over your files using vector embeddings.
//! This binary provides commands for indexing, querying, mounting, and
//! managing the search index.
//!
//! ## Commands
//!
//! - `ragfs index <PATH>` - Index a directory for semantic search
//! - `ragfs query <PATH> <QUERY>` - Search indexed content
//! - `ragfs mount <SOURCE> <MOUNTPOINT>` - Mount as FUSE filesystem
//! - `ragfs status <PATH>` - Show index statistics
//!
//! ## Examples
//!
//! ```bash
//! # Index a directory
//! ragfs index ~/Documents
//!
//! # Search for content
//! ragfs query ~/Documents "machine learning implementation"
//!
//! # Get JSON output
//! ragfs query ~/Documents "auth" --format json
//! ```
//!
//! See the [User Guide](../docs/USER_GUIDE.md) for detailed usage.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
#[cfg(feature = "mount")]
use daemonize::Daemonize;
use ragfs_chunker::{ChunkerRegistry, CodeChunker, FixedSizeChunker, SemanticChunker};
use ragfs_core::{ChunkConfig, Embedder, EmbeddingConfig, Indexer, VectorStore};
#[cfg(feature = "candle")]
use ragfs_embed::CandleEmbedder;
use ragfs_embed::EmbedderPool;
use ragfs_extract::{ExtractorRegistry, ImageExtractor, PdfExtractor, TextExtractor};
use ragfs_index::{IndexerConfig, IndexerService};
use ragfs_query::QueryExecutor;
#[cfg(feature = "lancedb")]
use ragfs_store::LanceStore;
use serde::Serialize;
#[cfg(feature = "mount")]
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

mod config;

use config::{Config, data_dir};

/// Embedding dimension for gte-small model.
const EMBEDDING_DIM: usize = 384;

#[derive(Parser)]
#[command(name = "ragfs")]
#[command(about = "A FUSE filesystem for RAG architectures")]
#[command(version)]
struct Cli {
    /// Path to config file (default: ~/.config/ragfs/config.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Mount a directory as a RAGFS filesystem
    #[cfg(feature = "mount")]
    Mount {
        /// Source directory to index
        source: PathBuf,

        /// Mount point
        mountpoint: PathBuf,

        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,

        /// Allow other users to access the mount
        #[arg(long)]
        allow_other: bool,
    },

    /// Index a directory (without mounting)
    Index {
        /// Directory to index
        path: PathBuf,

        /// Force reindexing of all files
        #[arg(short, long)]
        force: bool,

        /// Watch for changes after initial indexing
        #[arg(short, long)]
        watch: bool,
    },

    /// Query the index
    Query {
        /// Path to indexed directory
        path: PathBuf,

        /// Query string
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Use hybrid search (vector + full-text) instead of pure vector
        /// similarity. Experimental: the LanceDB FTS index is built on the
        /// empty table and not yet refreshed after inserts, and hybrid result
        /// scores are not surfaced correctly, so this is off by default until
        /// those are fixed.
        #[arg(long)]
        hybrid: bool,
    },

    /// Show index status
    Status {
        /// Path to indexed directory
        path: PathBuf,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Print sample configuration file
    Init,
    /// Show config file path
    Path,
}

/// Output structure for query results.
#[derive(Serialize)]
struct QueryOutput {
    query: String,
    results: Vec<ResultItem>,
}

#[derive(Serialize)]
struct ResultItem {
    file: String,
    score: f32,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    lines: Option<String>,
}

/// Output structure for status.
#[derive(Serialize)]
struct StatusOutput {
    path: String,
    total_files: u64,
    total_chunks: u64,
    index_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_updated: Option<String>,
}

/// Get the database path for a given source directory.
fn get_db_path(source: &PathBuf) -> Result<PathBuf> {
    let data = data_dir().context("Failed to get data directory")?;
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];
    Ok(data.join("indices").join(hash_str).join("index.lance"))
}

/// Path to the marker file recording which embedding model built an index.
fn embedding_model_marker_path(db_path: &std::path::Path) -> PathBuf {
    // db_path is `.../indices/{hash}/index.lance`; keep the marker beside it.
    let dir = db_path.parent().unwrap_or(db_path);
    dir.join("embedding_model")
}

/// Read the embedding model recorded for an index, if any.
fn read_embedding_model(marker: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(marker)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Record the embedding model that built an index.
fn write_embedding_model(marker: &std::path::Path, model: &str) -> std::io::Result<()> {
    if let Some(dir) = marker.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(marker, model)
}

/// Whether the recorded model differs from the current one. A missing marker
/// (fresh index) is not a change; only a *different* recorded model forces a
/// full reindex, because the stored embeddings are then incompatible.
fn embedding_model_changed(recorded: Option<&str>, current: &str) -> bool {
    matches!(recorded, Some(m) if m != current)
}

/// Get the PID file path for a mount.
///
/// Uses `$XDG_RUNTIME_DIR/ragfs/` if available, otherwise falls back to
/// `$XDG_CACHE_HOME/ragfs/run/`.
#[cfg(feature = "mount")]
fn get_pid_path(source: &PathBuf) -> Result<PathBuf> {
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];

    // Try XDG_RUNTIME_DIR first (e.g., /run/user/1000)
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(runtime_dir).join("ragfs");
        std::fs::create_dir_all(&dir).ok();
        return Ok(dir.join(format!("{hash_str}.pid")));
    }

    // Fallback to cache directory
    let dirs = directories::ProjectDirs::from("", "", "ragfs")
        .context("Failed to get project directories")?;
    let dir = dirs.cache_dir().join("run");
    std::fs::create_dir_all(&dir).context("Failed to create PID directory")?;
    Ok(dir.join(format!("{hash_str}.pid")))
}

/// Get the log file path for daemon output.
#[cfg(feature = "mount")]
fn get_log_path(source: &PathBuf) -> Result<PathBuf> {
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];

    let dirs = directories::ProjectDirs::from("", "", "ragfs")
        .context("Failed to get project directories")?;
    let dir = dirs.cache_dir().join("logs");
    std::fs::create_dir_all(&dir).context("Failed to create log directory")?;
    Ok(dir.join(format!("{hash_str}.log")))
}

/// Create the standard component stack.
async fn create_components(
    source: PathBuf,
) -> Result<(
    Arc<LanceStore>,
    Arc<ExtractorRegistry>,
    Arc<ChunkerRegistry>,
    Arc<EmbedderPool>,
)> {
    // Create store
    let db_path = get_db_path(&source)?;
    let store = Arc::new(LanceStore::new(db_path, EMBEDDING_DIM));

    // Create extractor registry
    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());
    extractors.register("pdf", PdfExtractor::new());
    extractors.register("image", ImageExtractor::new());
    let extractors = Arc::new(extractors);

    // Create chunker registry
    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.register("code", CodeChunker::new());
    chunkers.register("semantic", SemanticChunker::new());
    chunkers.set_default("fixed");
    let chunkers = Arc::new(chunkers);

    // Create embedder
    let cache_dir = data_dir()
        .context("Failed to get data directory")?
        .join("models");
    let embedder = CandleEmbedder::new(cache_dir);

    // Initialize embedder (downloads model if needed)
    info!("Initializing embedder (this may download the model on first run)...");
    embedder
        .init()
        .await
        .context("Failed to initialize embedder")?;

    let embedder_pool = Arc::new(EmbedderPool::new(
        Arc::new(embedder) as Arc<dyn Embedder>,
        4,
    ));

    Ok((store, extractors, chunkers, embedder_pool))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    match cli.command {
        #[cfg(feature = "mount")]
        Commands::Mount {
            source,
            mountpoint,
            foreground,
            allow_other,
        } => {
            info!("Mounting {:?} at {:?}", source, mountpoint);

            // Verify paths exist
            if !source.exists() {
                anyhow::bail!("Source directory does not exist: {}", source.display());
            }
            if !mountpoint.exists() {
                anyhow::bail!("Mount point does not exist: {}", mountpoint.display());
            }

            let source = source.canonicalize()?;

            // Create components for RAG functionality
            let (store, extractors, chunkers, embedder_pool) =
                create_components(source.clone()).await?;

            // Initialize store
            store.init().await.context("Failed to initialize store")?;

            // Create indexer for reindex requests
            let indexer_config = IndexerConfig {
                chunk_config: ChunkConfig::default(),
                embed_config: EmbeddingConfig::default(),
                ..Default::default()
            };

            let indexer = Arc::new(IndexerService::new(
                source.clone(),
                store.clone() as Arc<dyn VectorStore>,
                extractors,
                chunkers,
                embedder_pool.clone(),
                indexer_config,
            ));

            // Create channel for reindex requests
            let (reindex_tx, mut reindex_rx) = tokio::sync::mpsc::channel::<PathBuf>(32);

            // Spawn reindex handler task
            let indexer_clone = indexer.clone();
            let reindex_handler = tokio::spawn(async move {
                while let Some(path) = reindex_rx.recv().await {
                    info!("Processing reindex request for: {:?}", path);
                    match indexer_clone.reindex_path(&path).await {
                        Ok(()) => {
                            info!("Successfully reindexed: {:?}", path);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to reindex {:?}: {}", path, e);
                        }
                    }
                }
            });

            // Get runtime handle for FUSE (which runs in blocking context)
            let runtime = tokio::runtime::Handle::current();

            // Create filesystem with RAG capabilities
            let fs = ragfs_fuse::RagFs::with_rag(
                source.clone(),
                store as Arc<dyn VectorStore>,
                embedder_pool.document_embedder(),
                runtime,
                Some(reindex_tx),
            );

            // Build mount options
            let mut options = vec![
                fuser::MountOption::FSName("ragfs".to_string()),
                fuser::MountOption::AutoUnmount,
                fuser::MountOption::DefaultPermissions,
            ];

            if allow_other {
                options.push(fuser::MountOption::AllowOther);
            }

            // Mount
            if foreground {
                info!("Running in foreground (Ctrl+C to unmount)");
                info!("Try: cat {:?}/.ragfs/.index", mountpoint);
                info!(
                    "Reindex: echo 'path/to/file' > {:?}/.ragfs/.reindex",
                    mountpoint
                );
                fuser::mount2(fs, &mountpoint, &options)?;
            } else {
                // Daemonize: fork to background
                let pid_path = get_pid_path(&source)?;
                let log_path = get_log_path(&source)?;

                // Print info before daemonizing (these won't be visible after fork)
                println!("Mounting in background...");
                println!("PID file: {}", pid_path.display());
                println!("Log file: {}", log_path.display());
                println!("Try: cat {}/.ragfs/.index", mountpoint.display());
                println!("Unmount: fusermount -u {}", mountpoint.display());

                // Open log file for stdout/stderr redirection
                let stdout =
                    File::create(&log_path).context("Failed to create log file for stdout")?;
                let stderr =
                    File::create(&log_path).context("Failed to create log file for stderr")?;

                let daemonize = Daemonize::new()
                    .pid_file(&pid_path)
                    .chown_pid_file(true)
                    .working_directory("/")
                    .stdout(stdout)
                    .stderr(stderr);

                match daemonize.start() {
                    Ok(()) => {
                        // We're now in the daemon process
                        // Re-initialize tracing to log file since we've forked
                        fuser::mount2(fs, &mountpoint, &options)?;
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to daemonize: {e}");
                    }
                }
            }

            // Cleanup reindex handler on unmount
            reindex_handler.abort();
        }

        Commands::Index { path, force, watch } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;

            let (store, extractors, chunkers, embedder) = create_components(path.clone()).await?;

            // Detect an embedding-model change: the index stores model-specific
            // vectors, so if the recorded model differs from the current one the
            // whole index is stale and must be rebuilt.
            let db_path = get_db_path(&path)?;
            let marker = embedding_model_marker_path(&db_path);
            let current_model = embedder.model_name().to_string();
            let model_changed =
                embedding_model_changed(read_embedding_model(&marker).as_deref(), &current_model);
            if model_changed {
                info!(
                    "Embedding model changed to {current_model}; forcing a full reindex"
                );
            }
            let force = force || model_changed;
            info!("Indexing {:?} (force={})", path, force);

            // Create indexer config
            let config = IndexerConfig {
                chunk_config: ChunkConfig::default(),
                embed_config: EmbeddingConfig::default(),
                force,
                ..Default::default()
            };

            // Create indexer
            let indexer = IndexerService::new(
                path.clone(),
                store.clone() as Arc<dyn VectorStore>,
                extractors,
                chunkers,
                embedder,
                config,
            );

            // Subscribe to updates for progress
            let mut updates = indexer.subscribe();

            // Spawn progress reporter
            let progress_handle = tokio::spawn(async move {
                let mut indexed = 0u64;
                let mut errors = 0u64;
                while let Ok(update) = updates.recv().await {
                    match update {
                        ragfs_index::IndexUpdate::FileIndexed { path, chunk_count } => {
                            indexed += 1;
                            info!("Indexed: {:?} ({} chunks)", path, chunk_count);
                        }
                        ragfs_index::IndexUpdate::FileError { path, error } => {
                            errors += 1;
                            tracing::warn!("Error: {:?}: {}", path, error);
                        }
                        ragfs_index::IndexUpdate::IndexingStarted { .. } => {}
                        ragfs_index::IndexUpdate::FileRemoved { .. } => {}
                    }
                }
                (indexed, errors)
            });

            // Start indexer; `queued` is how many files the initial scan found.
            let queued = indexer.start().await.context("Failed to start indexer")?;

            if watch {
                // Record the model now (best effort) so a later run detects a swap.
                let _ = write_embedding_model(&marker, &current_model);
                info!("Watching for changes. Press Ctrl+C to stop.");
                // Wait indefinitely
                tokio::signal::ctrl_c()
                    .await
                    .context("Failed to wait for Ctrl+C")?;
                indexer.stop().await?;
            } else {
                // Wait for the initial scan's files to actually finish indexing,
                // rather than guessing with a fixed sleep. Each queued file emits
                // exactly one indexed/error result.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
                loop {
                    let s = indexer.stats().await.context("Failed to read index stats")?;
                    if (s.indexed_files + s.error_files) as usize >= queued {
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        tracing::warn!(
                            "Indexing did not finish within the time limit ({}/{} files)",
                            s.indexed_files + s.error_files,
                            queued
                        );
                        break;
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                }

                indexer.stop().await?;
                // Record the model that built this index (enables swap detection).
                write_embedding_model(&marker, &current_model)
                    .context("Failed to write embedding-model marker")?;

                let stats = store.stats().await?;
                info!(
                    "Indexing complete: {} files, {} chunks",
                    stats.total_files, stats.total_chunks
                );
            }

            drop(progress_handle);
        }

        Commands::Query {
            path,
            query,
            limit,
            hybrid,
        } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;

            // Check if index exists
            let db_path = get_db_path(&path)?;
            if !db_path.exists() {
                anyhow::bail!(
                    "Index not found for {}. Run 'ragfs index {}' first.",
                    path.display(),
                    path.display()
                );
            }

            let (store, _extractors, _chunkers, embedder) = create_components(path).await?;

            // Initialize store
            store.init().await.context("Failed to initialize store")?;

            // Create query executor
            let executor = QueryExecutor::new(
                store as Arc<dyn VectorStore>,
                embedder.document_embedder(),
                limit,
                hybrid, // opt-in; vector-only by default (see --hybrid help)
            );

            // Execute query
            let results = executor
                .execute(&query)
                .await
                .context("Query execution failed")?;

            // Output results
            match cli.format {
                OutputFormat::Json => {
                    let output = QueryOutput {
                        query: query.clone(),
                        results: results
                            .iter()
                            .map(|r| ResultItem {
                                file: r.file_path.to_string_lossy().to_string(),
                                score: r.score,
                                content: truncate(&r.content, 200),
                                lines: r
                                    .line_range
                                    .as_ref()
                                    .map(|l| format!("{}:{}", l.start, l.end)),
                            })
                            .collect(),
                    };
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Text => {
                    println!("Query: {query}\n");
                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        for (i, result) in results.iter().enumerate() {
                            println!(
                                "{}. {} (score: {:.3})",
                                i + 1,
                                result.file_path.display(),
                                result.score
                            );
                            if let Some(ref lines) = result.line_range {
                                println!("   Lines: {}-{}", lines.start, lines.end);
                            }
                            println!("   {}", truncate(&result.content, 100));
                            println!();
                        }
                    }
                }
            }
        }

        Commands::Status { path } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;
            let db_path = get_db_path(&path)?;

            if !db_path.exists() {
                match cli.format {
                    OutputFormat::Json => {
                        println!(r#"{{"error": "Index not found"}}"#);
                    }
                    OutputFormat::Text => {
                        println!("Index not found for {}", path.display());
                        println!("Run 'ragfs index {}' to create it.", path.display());
                    }
                }
                return Ok(());
            }

            let store = LanceStore::new(db_path, EMBEDDING_DIM);
            store.init().await.context("Failed to initialize store")?;

            let stats = store.stats().await?;

            match cli.format {
                OutputFormat::Json => {
                    let output = StatusOutput {
                        path: path.to_string_lossy().to_string(),
                        total_files: stats.total_files,
                        total_chunks: stats.total_chunks,
                        index_size_bytes: stats.index_size_bytes,
                        last_updated: stats.last_updated.map(|t| t.to_rfc3339()),
                    };
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Text => {
                    println!("Index Status for {}", path.display());
                    println!("  Files:  {}", stats.total_files);
                    println!("  Chunks: {}", stats.total_chunks);
                    if let Some(last) = stats.last_updated {
                        println!("  Updated: {}", last.format("%Y-%m-%d %H:%M:%S"));
                    }
                }
            }
        }

        Commands::Config { action } => {
            // Load config from file or CLI-specified path
            let config = if let Some(ref path) = cli.config {
                Config::load_from(Some(path.clone()))
                    .context(format!("Failed to load config from {}", path.display()))?
            } else {
                Config::load().context("Failed to load config")?
            };

            match action {
                ConfigAction::Show => match cli.format {
                    OutputFormat::Json => {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&config)
                                .context("Failed to serialize config")?
                        );
                    }
                    OutputFormat::Text => {
                        println!(
                            "{}",
                            toml::to_string_pretty(&config)
                                .context("Failed to serialize config")?
                        );
                    }
                },
                ConfigAction::Init => {
                    println!("{}", Config::sample_toml());
                }
                ConfigAction::Path => {
                    if let Some(path) = Config::config_path() {
                        println!("{}", path.display());
                    } else {
                        println!("Could not determine config directory");
                    }
                }
            }
        }
    }

    Ok(())
}

/// Truncate a string to max length, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model_changed() {
        // Fresh index (no marker): not a change, don't force.
        assert!(!embedding_model_changed(None, "intfloat/multilingual-e5-small"));
        // Same model: not a change.
        assert!(!embedding_model_changed(
            Some("intfloat/multilingual-e5-small"),
            "intfloat/multilingual-e5-small"
        ));
        // Different model (the gte -> e5 swap): change, must force reindex.
        assert!(embedding_model_changed(
            Some("thenlper/gte-small"),
            "intfloat/multilingual-e5-small"
        ));
    }

    #[test]
    fn test_embedding_model_marker_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("indices/abc/index.lance");
        let marker = embedding_model_marker_path(&db_path);

        // Nothing recorded yet.
        assert_eq!(read_embedding_model(&marker), None);

        write_embedding_model(&marker, "intfloat/multilingual-e5-small").unwrap();
        assert_eq!(
            read_embedding_model(&marker).as_deref(),
            Some("intfloat/multilingual-e5-small")
        );
        // Marker sits beside the index, not inside index.lance.
        assert_eq!(marker.parent(), db_path.parent());
    }
}
