//! Multilingual e5-small embedder using Candle.
//!
//! Uses intfloat/multilingual-e5-small for text embeddings:
//! - 384 dimensions
//! - 512 max tokens
//! - BERT architecture, multilingual (EN/ZH/...)
//! - Asymmetric prefixes: documents are encoded as `passage: ...`, queries as
//!   `query: ...`, which is what gives e5 its retrieval discrimination.

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Cache, Repo, RepoType, api::tokio::Api};
use ragfs_core::{EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality};
use std::panic;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
#[cfg(all(target_os = "macos", feature = "metal"))]
use tracing::warn;
use tracing::{debug, info};

/// Model identifier on `HuggingFace` Hub.
const MODEL_ID: &str = "intfloat/multilingual-e5-small";

/// Embedding dimension for multilingual-e5-small.
const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length.
const MAX_TOKENS: usize = 512;

/// Prefix e5 expects on query text.
const QUERY_PREFIX: &str = "query: ";

/// Prefix e5 expects on document/passage text.
const PASSAGE_PREFIX: &str = "passage: ";

fn model_repo() -> Repo {
    Repo::new(MODEL_ID.to_string(), RepoType::Model)
}

fn cached_model_file(cache: &Cache, repo: &Repo, filename: &str) -> Option<PathBuf> {
    cache.repo(repo.clone()).get(filename)
}

fn create_hf_api() -> Result<Api, EmbedError> {
    match panic::catch_unwind(Api::new) {
        Ok(Ok(api)) => Ok(api),
        Ok(Err(err)) => Err(EmbedError::ModelLoad(format!(
            "Failed to create HuggingFace API: {err}"
        ))),
        Err(_) => Err(EmbedError::ModelLoad(
            "Failed to create HuggingFace API: client initialization panicked".to_string(),
        )),
    }
}

async fn resolve_model_file(
    cache: &Cache,
    api: &mut Option<Api>,
    repo: &Repo,
    filename: &str,
) -> Result<PathBuf, EmbedError> {
    if let Some(path) = cached_model_file(cache, repo, filename) {
        debug!("Using cached model file: {:?}", path);
        return Ok(path);
    }

    if api.is_none() {
        *api = Some(create_hf_api()?);
    }

    let api_repo = api
        .as_ref()
        .expect("HuggingFace API should be initialized")
        .repo(repo.clone());

    debug!("Downloading model file: {filename}");
    api_repo
        .get(filename)
        .await
        .map_err(|e| EmbedError::ModelLoad(format!("Failed to resolve {filename}: {e}")))
}

/// Build the model input for a query (e5 asymmetric prefix).
fn query_input(text: &str) -> String {
    format!("{QUERY_PREFIX}{text}")
}

/// Build the model input for a document/passage (e5 asymmetric prefix).
fn passage_input(text: &str) -> String {
    format!("{PASSAGE_PREFIX}{text}")
}

/// Multilingual e5-small embedder using Candle.
pub struct CandleEmbedder {
    /// Device to run inference on (Metal, CUDA, or CPU)
    device: Device,
    /// Loaded model
    model: Arc<RwLock<Option<BertModel>>>,
    /// Tokenizer
    tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    /// Model configuration
    config: Arc<RwLock<Option<Config>>>,
    /// Cache directory for models
    #[allow(dead_code)]
    cache_dir: PathBuf,
    /// Whether model is initialized
    initialized: Arc<RwLock<bool>>,
}

impl CandleEmbedder {
    /// Create a new `CandleEmbedder`.
    pub fn new(cache_dir: PathBuf) -> Self {
        let device = default_device();
        info!("CandleEmbedder using device: {:?}", device);

        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Create with specific device.
    pub fn with_device(cache_dir: PathBuf, device: Device) -> Self {
        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize the model (download if needed, load into memory).
    pub async fn init(&self) -> Result<(), EmbedError> {
        {
            let initialized = self.initialized.read().await;
            if *initialized {
                return Ok(());
            }
        }

        info!("Initializing CandleEmbedder with model: {}", MODEL_ID);

        let cache = Cache::from_env();
        let repo = model_repo();
        let mut api = None;

        let tokenizer_path = resolve_model_file(&cache, &mut api, &repo, "tokenizer.json").await?;
        let config_path = resolve_model_file(&cache, &mut api, &repo, "config.json").await?;
        let weights_path = resolve_model_file(&cache, &mut api, &repo, "model.safetensors").await?;

        // Load tokenizer
        debug!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;

        // Load config
        debug!("Loading config...");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to read config: {e}")))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to parse config: {e}")))?;

        // Load model weights
        debug!("Loading model weights...");
        // SAFETY: The safetensors file is downloaded from HuggingFace Hub and is trusted.
        // Memory mapping is safe for read-only access to model weights.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &self.device)
                .map_err(|e| EmbedError::ModelLoad(format!("Failed to load weights: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to create BERT model: {e}")))?;

        // Store in instance
        {
            let mut tok = self.tokenizer.write().await;
            *tok = Some(tokenizer);
        }
        {
            let mut cfg = self.config.write().await;
            *cfg = Some(config);
        }
        {
            let mut mdl = self.model.write().await;
            *mdl = Some(model);
        }
        {
            let mut init = self.initialized.write().await;
            *init = true;
        }

        info!("CandleEmbedder initialized successfully");
        Ok(())
    }

    /// Mean pooling with attention mask.
    fn mean_pooling(
        &self,
        token_embeddings: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, EmbedError> {
        // Expand attention mask to match embedding dimensions
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| EmbedError::Inference(format!("unsqueeze failed: {e}")))?
            .broadcast_as(token_embeddings.shape())
            .map_err(|e| EmbedError::Inference(format!("broadcast failed: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| EmbedError::Inference(format!("dtype conversion failed: {e}")))?;

        // Masked sum
        let masked = token_embeddings
            .mul(&mask)
            .map_err(|e| EmbedError::Inference(format!("mul failed: {e}")))?;

        let sum = masked
            .sum(1)
            .map_err(|e| EmbedError::Inference(format!("sum failed: {e}")))?;

        // Count non-masked tokens
        let mask_sum = mask
            .sum(1)
            .map_err(|e| EmbedError::Inference(format!("mask sum failed: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| EmbedError::Inference(format!("clamp failed: {e}")))?;

        // Mean
        let mean = sum
            .div(&mask_sum)
            .map_err(|e| EmbedError::Inference(format!("div failed: {e}")))?;

        Ok(mean)
    }

    /// L2 normalize embeddings.
    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor, EmbedError> {
        let norm = embeddings
            .sqr()
            .map_err(|e| EmbedError::Inference(format!("sqr failed: {e}")))?
            .sum_keepdim(1)
            .map_err(|e| EmbedError::Inference(format!("sum_keepdim failed: {e}")))?
            .sqrt()
            .map_err(|e| EmbedError::Inference(format!("sqrt failed: {e}")))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| EmbedError::Inference(format!("clamp failed: {e}")))?;

        let normalized = embeddings
            .broadcast_div(&norm)
            .map_err(|e| EmbedError::Inference(format!("div failed: {e}")))?;

        Ok(normalized)
    }

    /// Encode a batch of texts.
    async fn encode_batch(
        &self,
        texts: &[&str],
        normalize: bool,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        // Ensure initialized
        self.init().await?;

        let tokenizer = self.tokenizer.read().await;
        let tokenizer = tokenizer
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Tokenizer not loaded".to_string()))?;

        let model = self.model.read().await;
        let model = model
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Model not loaded".to_string()))?;

        // Tokenize all texts
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Inference(format!("Tokenization failed: {e}")))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(0);
        let max_len = max_len.min(MAX_TOKENS);

        // Prepare input tensors
        let mut input_ids_vec: Vec<u32> = Vec::new();
        let mut attention_mask_vec: Vec<u32> = Vec::new();
        let mut token_type_ids_vec: Vec<u32> = Vec::new();
        let mut token_counts = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len);
            token_counts.push(len);

            // Add IDs with padding
            for i in 0..max_len {
                if i < len {
                    input_ids_vec.push(ids[i]);
                    attention_mask_vec.push(1);
                    token_type_ids_vec.push(0);
                } else {
                    input_ids_vec.push(0); // PAD token
                    attention_mask_vec.push(0);
                    token_type_ids_vec.push(0);
                }
            }
        }

        let batch_size = texts.len();

        // Create tensors
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)
            .map_err(|e| {
                EmbedError::Inference(format!("Failed to create input_ids tensor: {e}"))
            })?;

        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device).map_err(
                |e| EmbedError::Inference(format!("Failed to create attention_mask tensor: {e}")),
            )?;

        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device).map_err(
                |e| EmbedError::Inference(format!("Failed to create token_type_ids tensor: {e}")),
            )?;

        // Run model
        let output = model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| EmbedError::Inference(format!("Model forward failed: {e}")))?;

        // Mean pooling
        let pooled = self.mean_pooling(&output, &attention_mask)?;

        // Normalize if requested
        let final_embeddings = if normalize {
            self.normalize(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<EmbeddingOutput>
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let embedding = final_embeddings
                .get(i)
                .map_err(|e| EmbedError::Inference(format!("Failed to get embedding {i}: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| EmbedError::Inference(format!("Failed to convert to vec: {e}")))?;

            results.push(EmbeddingOutput {
                embedding,
                token_count: token_counts[i],
            });
        }

        Ok(results)
    }
}

fn default_device() -> Device {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        match try_new_metal_device(0) {
            Ok(Ok(device)) => return device,
            Ok(Err(err)) => warn!("Metal device unavailable, falling back to CPU/CUDA: {err}"),
            Err(_) => warn!("Metal device initialization panicked, falling back to CPU/CUDA"),
        }
    }

    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn try_new_metal_device(ordinal: usize) -> std::thread::Result<candle_core::Result<Device>> {
    static METAL_INIT_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    let _guard = METAL_INIT_HOOK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(|| Device::new_metal(ordinal));
    panic::set_hook(previous_hook);
    result
}

#[async_trait]
impl Embedder for CandleEmbedder {
    fn model_name(&self) -> &str {
        MODEL_ID
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }

    fn max_tokens(&self) -> usize {
        MAX_TOKENS
    }

    fn modalities(&self) -> &[Modality] {
        &[Modality::Text]
    }

    async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Embedding {} texts with batch_size {}",
            texts.len(),
            config.batch_size
        );

        // e5 expects documents to be prefixed with "passage: ".
        let prefixed: Vec<String> = texts.iter().map(|t| passage_input(t)).collect();

        // Process in batches
        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in prefixed.chunks(config.batch_size) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let batch_results = self.encode_batch(&refs, config.normalize).await?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        // e5 expects queries to be prefixed with "query: " (asymmetric to the
        // "passage: " prefix used for documents in embed_text). We must encode
        // directly rather than via embed_text, which would apply the passage
        // prefix instead.
        let input = query_input(query);
        let results = self
            .encode_batch(&[input.as_str()], config.normalize)
            .await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Inference("Empty embedding result".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_is_multilingual() {
        // The model must be multilingual so bilingual (EN/ZH) vaults retrieve
        // sensibly; gte-small was English-only.
        assert!(
            MODEL_ID.contains("multilingual-e5"),
            "expected a multilingual-e5 model, got {MODEL_ID}"
        );
    }

    #[test]
    fn test_e5_prefixes() {
        // e5 models are trained with asymmetric prefixes: documents are encoded
        // as "passage: ..." and queries as "query: ...". Applying them is what
        // gives e5 its retrieval discrimination.
        assert_eq!(passage_input("hello"), "passage: hello");
        assert_eq!(query_input("what is x"), "query: what is x");
    }

    #[test]
    fn test_cached_model_file_uses_huggingface_cache() {
        let cache_dir = tempdir().unwrap();
        let cache = Cache::new(cache_dir.path().to_path_buf());
        let repo = model_repo();
        let commit = "abc123";

        cache.repo(repo.clone()).create_ref(commit).unwrap();

        let path = cache_dir
            .path()
            .join(repo.folder_name())
            .join("snapshots")
            .join(commit)
            .join("tokenizer.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}").unwrap();

        assert_eq!(
            cached_model_file(&cache, &repo, "tokenizer.json").as_deref(),
            Some(path.as_path())
        );
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_candle_embedder() {
        let cache_dir = tempdir().unwrap();
        let embedder = CandleEmbedder::new(cache_dir.path().to_path_buf());

        embedder.init().await.unwrap();

        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_name(), "intfloat/multilingual-e5-small");

        let config = EmbeddingConfig::default();
        let texts = &["Hello world", "This is a test"];

        let results = embedder.embed_text(texts, &config).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].embedding.len(), 384);
        assert_eq!(results[1].embedding.len(), 384);

        // Check normalization (should have unit length)
        let norm: f32 = results[0]
            .embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
