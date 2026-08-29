# Retrieval Core Improvements — Design

**Date:** 2026-08-29
**Status:** Approved (design), pending implementation
**Context:** Making ragfs usable for semantic search over a real, mixed-language
(EN/ZH) Obsidian vault containing markdown, PDFs, and image attachments.

## Problem

Evaluated ragfs on a real Obsidian vault (`03_Resources`, 182 files / 441
chunks, clean markdown). Retrieval quality was unusable. Root causes found:

1. **Exclusion is broken.** `IndexerConfig` ships sensible `exclude_patterns`
   (`**/.*`, `**/.git/**`, …) but the hand-written glob matcher in
   `scan_directory` (crates/ragfs-index/src/indexer.rs) is a toy: it splits on
   `**` and does literal substring checks. Pattern `**/.*` becomes "path
   contains the literal substring `.*`" (never true), and any pattern with two
   `**` segments (e.g. `**/.git/**`) splits into 3 parts and returns `false`.
   Result: `.obsidian/*.json`, `.git`, and all dotfiles get indexed. These
   short, near-identical config blobs score 0.93–0.98 against every query and
   dominate results.

2. **Image placeholder pollution.** Without the `vision` feature,
   `ImageExtractor` (crates/ragfs-extract/src/image.rs) returns a non-empty
   placeholder like `"3072x4080 jpeg image"`. This passes the existing
   empty-content skip in `process_file`, so every png/jpg is indexed as a
   near-identical short string and floods every query at ~0.9.

3. **Hybrid search is disabled.** `main.rs` constructs `QueryExecutor` with
   `hybrid = false` (hardcoded), so retrieval is pure vector search even though
   the store implements `hybrid_search` (vector + full-text BM25). Keyword-ish
   queries (e.g. "milky way") lose to the short-note attractor because there is
   no lexical signal to rescue the correct document.

4. **English-only embedding model.** The model is `thenlper/gte-small` (384d,
   English). The target vault is bilingual EN/ZH; Chinese semantic recall is
   weak. (Embeddings *are* L2-normalized — verified — so the short-note
   attractor is inherent small-model/mean-pooling behavior, not a normalization
   bug.)

## Non-goals

- FUSE mount changes (already handled: optional `mount` feature).
- Building an Obsidian plugin / UI (separate sub-project).
- PDF extraction robustness for CN/scanned PDFs (separate sub-project).
- Cross-encoder reranking (deferred; revisit only if 1–4 are insufficient).

## Design

Four changes, ordered cheapest-and-highest-leverage first. Re-test on
`03_Resources` after each, against the known baseline (correct note
`Photography/Milky way.md` buried out of top-15; `Vault/Rowenta …` — a note
whose whole body is `XD6520F0` — tops every query).

### 1. Fix exclusion (required)

Replace the hand-written glob in `scan_directory` and `reindex_directory` with a
real matcher: an `ExcludeMatcher` that compiles `exclude_patterns` via `globset`
(correct glob semantics) plus a hidden-file rule (any path component starting
with `.`), which is what a correct `**/.*` would do. Exclusion is evaluated on
the path *relative to the index root*, so a vault that itself lives under a
hidden directory (e.g. `~/.notes/vault`) is not wholly excluded. Directory
pruning happens at the directory level (excluded dirs are not descended into).

Implemented with `globset` rather than the `ignore` crate's `WalkBuilder`: the
two existing recursive walkers are kept, only the matcher is replaced, which is
the smaller, more contained change and keeps the matcher unit-testable. As a
result, `.gitignore`/`.ignore` are **not** consulted and there is **no**
`.ragfsignore` support — the hidden-file rule plus `exclude_patterns` covers the
`.obsidian`/`.git`/dotfile pollution this fix targets. (A future change could
adopt `WalkBuilder` if gitignore-awareness becomes desirable.)

**Acceptance:** indexing `03_Resources` indexes zero files under any
`.obsidian/` directory and zero dotfiles; `ragfs status` file count drops
accordingly. A vault rooted under a hidden directory still indexes its files.

### 2. Skip images when no OCR (required)

When the `vision` feature is off, `ImageExtractor` returns empty text (or the
extractor registry does not register it), so images fall through the existing
empty-content skip and are not indexed. No image entries in the store.

**Acceptance:** no `.png/.jpg/.jpeg` paths appear in any query result on a
photo-containing folder.

### 3. Enable hybrid search (required)

Make hybrid configurable and default it to on. Thread a `hybrid` flag from CLI
/ config into the `QueryExecutor` construction in `main.rs` (both the query and
mount paths), replacing the hardcoded `false`. Default `true`; allow
`--no-hybrid` (or a config key) to force pure vector.

**Acceptance:** for query "milky way", `Photography/Milky way.md` ranks in the
top 3; the `XD6520F0` note no longer tops unrelated queries.

### 4. Multilingual embedding model (evaluate after 1–3)

Only if 1–3 leave Chinese recall inadequate. Swap `MODEL_ID` in
`crates/ragfs-embed/src/candle.rs` to a multilingual model
(`intfloat/multilingual-e5-small`, 384d — dimension-compatible — or `bge-m3`).
Note: e5 models require `query:` / `passage:` prefixes, so the embedder must
apply the right prefix for query vs. document encoding (there is already a
`embed_query` path distinct from document embedding). Changing the model
invalidates existing indices — document that a reindex is required, and keep the
model id in one constant. If dimensions change from 384, `EMBEDDING_DIM` in
`main.rs` and the store schema must match.

**Acceptance:** on a set of Chinese-language queries against Chinese notes,
the correct note ranks in the top 3.

## Testing / Verification

- Unit: exclusion matcher (dotfiles, nested `.obsidian`, `.git`, custom
  `.ragfsignore`, multi-`**` globs) — table-driven tests replacing the current
  ones that assert on the broken matcher.
- Unit: image extractor returns empty text without `vision`.
- Integration/manual: reindex `~/ObsidianVault/03_Resources` after each change
  and compare the fixed query set's rankings to the recorded baseline.

## Rollout

Feature branch `retrieval-core-improvements`. Each of 1–3 is an independent
commit; 4 is a separate commit gated on evaluation. Reindex required after 1, 2,
and 4 (they change what/how content is stored).
