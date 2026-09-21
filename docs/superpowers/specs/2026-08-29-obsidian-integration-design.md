# Obsidian Integration — Design

**Date:** 2026-08-29
**Status:** Approved (design), pending implementation
**Depends on:** `retrieval-core-improvements` (e5 model, query fixes)

## Goal

Semantic search over an Obsidian vault, **from inside Obsidian**: type a query in a
sidebar panel, get ranked notes with snippets, click to open. Backed by ragfs's
local embeddings (fully offline).

## Architecture

Two pieces, split by ecosystem:

### Part A — `ragfs serve` (Rust, this repo)

A long-running local HTTP query server so the embedding model is loaded **once**
(the CLI `query` reloads it per invocation, ~0.5–1s — too slow for an
interactive search box).

- New subcommand: `ragfs serve <PATH> [--port 7777] [--host 127.0.0.1] [--limit 10]`.
- On start: open the LanceDB index for `<PATH>` (must already be indexed via
  `ragfs index`) and initialize the embedder once; hold them in shared state.
- Endpoint: `GET /query?q=<text>&limit=<n>` → JSON
  `{ "query": ..., "results": [ { "file", "score", "content", "lines"? } ] }`,
  reusing `QueryExecutor` (vector search; `--hybrid` off).
- Endpoint: `GET /health` → `{ "status": "ok", "model": ..., "path": ... }`.
- Binds **127.0.0.1 only** by default (local, no auth). CORS: allow the
  `app://obsidian.md` origin (and localhost) so the plugin's fetch works.
- **Query-only.** Keeping the index fresh stays a separate `ragfs index --watch
  <PATH>`. (Rationale: smaller MVP, clear separation; a future `--watch` flag on
  serve could fold indexing in.)
- HTTP via `axum`. Server is part of the default build (no new feature flag).

**Acceptance:** with a vault indexed, `ragfs serve <vault>` then
`curl 'http://127.0.0.1:7777/query?q=...'` returns ranked JSON results; a second
query is fast (model already warm).

### Part B — Obsidian plugin (TypeScript, separate project `~/Projects/ragfs-obsidian`)

- Settings: server URL (default `http://127.0.0.1:7777`).
- A right-sidebar `ItemView` "ragfs search": a text input; on Enter, `fetch`
  `/query`; render results as a list of (note basename, snippet, score).
- Clicking a result opens that note in the workspace (resolve the absolute path
  returned by ragfs to a vault-relative `TFile`).
- A command-palette command to reveal the search view.
- **Out of scope for MVP:** inserting `[[wikilink]]` at cursor (nice-to-have,
  add after the search+open loop works); indexing controls (user runs
  `ragfs index --watch` themselves); auth.

**Acceptance:** in Obsidian, typing a query in the panel shows the same ranked
notes as the CLI, and clicking one opens it.

## Non-goals

- Bundling/managing the ragfs binary from the plugin (user runs `serve`).
- Editing/agent file operations (that is the FUSE/MCP surface, not this).
- Publishing to the Obsidian community plugin list.

## Testing

- Rust: unit-test the request/response DTOs and query-param handling (limit
  default/clamp, empty query → 400). The full HTTP path is verified with `curl`
  against a real index (model-dependent, not unit-tested).
- Plugin: manual verification in Obsidian against a running `ragfs serve`.

## Rollout

- Part A on branch `obsidian-integration` in this repo (independent commits).
- Part B in a new `~/Projects/ragfs-obsidian` project with its own git repo.
