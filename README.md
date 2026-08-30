# RAGFS

[![CI](https://github.com/sophie4869/ragfs/actions/workflows/ci.yml/badge.svg)](https://github.com/sophie4869/ragfs/actions/workflows/ci.yml)
[![Security Audit](https://github.com/sophie4869/ragfs/actions/workflows/security.yml/badge.svg)](https://github.com/sophie4869/ragfs/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

Local semantic search over your files. Index a directory with fully-offline
embeddings (multilingual, so mixed English/Chinese corpora work) and query it by
meaning from the CLI or the local web server — `index`, `query`, `serve`,
`status`. On Linux it can additionally mount the index as a FUSE filesystem for
agent file operations (undo, audit, auto-organization); that piece is an
optional build feature and is not required for search.

> **Platforms:** the CLI and web server (`index`/`query`/`serve`/`status`) build
> and run on macOS and Linux. FUSE mounting is Linux-only and off by default — enable it with
> `--features mount` (needs `libfuse`).

## Features

- **Agent File Operations** - Structured file ops with JSON feedback via `.ops/` interface
- **Safety Layer** - Soft delete, audit logging, and undo support via `.safety/`
- **AI-Powered Management** - Auto-organization, deduplication, and cleanup via `.semantic/`
- **Semantic Search** - Query files by meaning using vector similarity search
- **Mobile Web Search** - Serve a private browser UI and JSON API for search/read/preview
- **Local Embeddings** - Runs entirely offline using the `multilingual-e5-small` model via Candle (multilingual, incl. Chinese)
- **FUSE Integration** *(Linux, optional `mount` feature)* - Mount indexed directories as a virtual filesystem
- **Real-time Indexing** - Watch directories for changes and update the index automatically
- **Multimodal Support** - Extract content from text, code, markdown, PDF, and images
- **Code-aware Chunking** - Syntax-aware splitting using tree-sitter for source code
- **Hybrid Search** *(experimental)* - Combine vector similarity with full-text search
- **MCP Server** - Claude Desktop integration for AI assistants

## Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| CLI (index, query, status) | Stable | Core functionality; builds on macOS and Linux |
| Web/API server | Beta | `ragfs serve`; mobile browser UI plus JSON/raw file endpoints |
| FUSE mount | Stable | Linux only; optional `mount` build feature |
| Semantic search | Stable | Vector similarity with LanceDB |
| Hybrid search | Experimental | Vector + full-text; opt-in via `--hybrid` (FTS wiring being fixed) |
| Text extraction | Stable | 40+ formats |
| Code chunking | Stable | Tree-sitter based |
| PDF extraction | Stable | Text + embedded images |
| Agent operations (.ops/) | Stable | JSON feedback, batch support |
| Safety layer (.safety/) | Stable | Trash, history, undo |
| Semantic operations (.semantic/) | Beta | Organize, dedupe, cleanup |
| Python bindings | Beta | PyO3 based |
| MCP server | Beta | Claude Desktop integration |
| Image captioning | Experimental | Optional, requires `vision` feature |

## Use Cases

**Ideal for:**
- LLM agents managing files (Claude, GPT, local models)
- Automated file organization and cleanup
- Safe file operations with audit trail
- Code repositories (1K-50K files)
- Documentation collections
- Research notes and papers
- Local-first semantic search

**Limitations:**
- FUSE mounting is Linux only (optional `mount` feature); CLI search runs on macOS and Linux
- Embedding model is downloaded on first run (~120MB) and cached
- Large repositories (100K+ files) may need tuning
- Hybrid (vector + full-text) search is experimental and opt-in via `ragfs query --hybrid`; default is vector-only

## Requirements

- Rust 1.88 or later
- `protoc` (Protocol Buffers compiler) for the LanceDB build dependency
  (`brew install protobuf` on macOS, `apt install protobuf-compiler` on Debian/Ubuntu)
- ~120MB disk for the embedding model (downloaded on first run, then cached)
- **For FUSE mount only (Linux):** `libfuse` (`libfuse-dev` on Debian/Ubuntu, `fuse` on Arch)

## Installation

```bash
# Clone the repository (this fork)
git clone https://github.com/sophie4869/ragfs.git
cd ragfs

# Build the CLI in release mode (index/query/status; no FUSE) — macOS & Linux.
# Build only the `ragfs` crate; a bare `cargo build` builds the whole workspace,
# including the Linux-only FUSE and PyO3 crates.
cargo build -p ragfs --release

# On Linux, to also build FUSE mounting:
#   cargo build -p ragfs --release --features mount

# Install to ~/.cargo/bin
cargo install --path crates/ragfs
```

## Quick Start

### Index a directory

```bash
# Index all files in a directory
ragfs index ~/Documents

# Watch for changes (continuous indexing)
ragfs index ~/Documents --watch
```

### Search your files

```bash
# Semantic search
ragfs query ~/Documents "machine learning implementation"

# Get more results
ragfs query ~/Documents "authentication logic" --limit 20

# JSON output for scripting (global flags precede the subcommand)
ragfs --format json query ~/Documents "database connection"
```

Search is vector-only by default. Hybrid (vector + full-text) is experimental
and opt-in: `ragfs query --hybrid ~/Documents "..."`.

### Search and read from a browser

```bash
ragfs serve ~/Documents --host 127.0.0.1 --port 7777
```

Open <http://127.0.0.1:7777> for the mobile-friendly web UI. The server also
exposes:

```text
GET /api/search?q=<text>&limit=<n>
GET /api/status
GET /api/files/<relative-path>
GET /raw/<relative-path>
```

For cross-language or personal vocabulary, add a `.ragfsaliases` file at the
indexed root (or an ancestor). It is read when `ragfs serve` starts and does not
require rebuilding the index:

```text
房东 = landlord, tenant, lease, rent
纠纷 = dispute, claim, conflict
证据 = evidence, proof, receipt, invoice
```

For a reverse-proxied or NAS deployment, keep `ragfs serve` on localhost or an
internal network and put authentication in front of it. You can also require a
bearer token:

```bash
RAGFS_SERVE_TOKEN="$(openssl rand -base64 32)" ragfs serve ~/Documents
```

### Mount as a filesystem (Linux only, requires `--features mount`)

```bash
# Create a mount point
mkdir ~/ragfs-mount

# Mount the indexed directory
ragfs mount ~/Documents ~/ragfs-mount --foreground
```

### Check index status

```bash
ragfs status ~/Documents
```

### Agent file operations (via FUSE mount)

```bash
# Create a file with feedback
echo -e "docs/new.md\n# New Document" > ~/ragfs-mount/.ragfs/.ops/.create
cat ~/ragfs-mount/.ragfs/.ops/.result  # JSON with undo_id

# Delete a file (soft delete to trash)
echo "docs/old.md" > ~/ragfs-mount/.ragfs/.ops/.delete

# Find similar files
echo "src/main.rs" > ~/ragfs-mount/.ragfs/.semantic/.similar
cat ~/ragfs-mount/.ragfs/.semantic/.similar

# Undo an operation
echo "<undo_id>" > ~/ragfs-mount/.ragfs/.safety/.undo
```

## CLI Reference

```
ragfs [OPTIONS] <COMMAND>

Commands:
  index   Index a directory
  query   Query the index
  status  Show index status
  config  Manage configuration
  mount   Mount a directory as a RAGFS filesystem   (only in --features mount builds; Linux)

Options:
  -c, --config <FILE>    Config file path [default: ~/.config/ragfs/config.toml]
  -v, --verbose          Enable verbose logging
  -f, --format <FORMAT>  Output format: text, json [default: text]  (global; precede the subcommand)
  -h, --help             Print help
  -V, --version          Print version
```

The default build ships `index`, `query`, `status`, and `config`. `mount`
appears only when built with `--features mount` (Linux, requires `libfuse`).

### mount

```
ragfs mount <SOURCE> <MOUNTPOINT> [OPTIONS]

Arguments:
  <SOURCE>      Source directory to index
  <MOUNTPOINT>  Mount point

Options:
  -f, --foreground  Run in foreground (don't daemonize)
      --allow-other Allow other users to access the mount
```

### index

```
ragfs index <PATH> [OPTIONS]

Arguments:
  <PATH>  Directory to index

Options:
  -f, --force  Force reindexing of all files
  -w, --watch  Watch for changes after initial indexing
```

### query

```
ragfs query <PATH> <QUERY> [OPTIONS]

Arguments:
  <PATH>   Path to indexed directory
  <QUERY>  Query string

Options:
  -l, --limit <LIMIT>  Maximum results [default: 10]
```

### status

```
ragfs status <PATH>

Arguments:
  <PATH>  Path to indexed directory
```

### config

```
ragfs config <ACTION>

Actions:
  show  Display current configuration
  init  Print sample config file
  path  Print config file path
```

## Architecture

RAGFS is organized as a Rust workspace with specialized crates:

| Crate | Description |
|-------|-------------|
| `ragfs` | CLI application |
| `ragfs-core` | Core traits and types |
| `ragfs-fuse` | FUSE filesystem implementation |
| `ragfs-index` | File indexing engine |
| `ragfs-chunker` | Document chunking strategies |
| `ragfs-embed` | Embedding generation (Candle) |
| `ragfs-extract` | Content extraction |
| `ragfs-store` | Vector storage (LanceDB) |
| `ragfs-query` | Query execution |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - 5-minute tutorial
- [User Guide](docs/USER_GUIDE.md) - Complete CLI reference
- [Configuration](docs/CONFIGURATION.md) - All config options
- [Performance Guide](docs/PERFORMANCE.md) - Tuning and optimization
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Architecture](docs/ARCHITECTURE.md) - Technical deep-dive
- [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md) - Why we made these choices
- [API Reference](docs/API.md) - Library usage and types
- [Python Bindings](docs/PYTHON.md) - Python SDK and framework integrations
- [MCP Server](docs/MCP.md) - Claude Desktop integration
- [Development Guide](docs/DEVELOPMENT.md) - Contributing to RAGFS

## How It Works

1. **Extraction** - Content is extracted from files based on their MIME type
2. **Chunking** - Text is split into overlapping chunks (~512 tokens each); near-empty chunks (frontmatter fences, lone headers) are dropped
3. **Embedding** - Each chunk is converted to a 384-dimensional vector using the `multilingual-e5-small` model (documents as `passage:`, queries as `query:`)
4. **Storage** - Vectors are stored in LanceDB for efficient similarity search. The embedding model is recorded alongside the index; changing it triggers a full reindex.
5. **Search** - Queries are embedded and matched against stored vectors using cosine similarity

## Storage Locations

- **Indices**: `~/.local/share/ragfs/indices/{hash}/index.lance` (macOS: `~/Library/Application Support/ragfs/indices/...`)
- **Models**: `~/.local/share/ragfs/models/` (macOS: `~/Library/Application Support/ragfs/models/`)
- **Embedding-model marker**: `embedding_model` file beside each index (used to detect model changes)

## Upstream & Acknowledgements

This is a fork of [RAGFS by Venere Labs](https://github.com/Venere-Labs/ragfs).
Changes in this fork focus on making the CLI usable on macOS and on real,
mixed-language note vaults: FUSE mounting made an optional build feature so the
CLI builds without `libfuse`; correct file exclusion; a multilingual embedding
model (`multilingual-e5-small`) with e5 `query:`/`passage:` prefixes; degenerate
/ empty-content chunk filtering; a working `--force`; and reindex-on-model-change
via an index marker. All credit for the original design and the bulk of the
implementation belongs to the upstream authors.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. Original work © Venere Labs; fork modifications under the same dual license.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
