# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-20

### Added

- **python**: Add PyO3 bindings for RAGFS
- **python**: Extend bindings with safety, semantic, and ops modules
- **examples**: Add LlamaIndex and Haystack RAG pipeline examples
- **extract**: Extract text from Office OOXML and ODT ([#55](https://github.com/sophie4869/ragfs/pull/55))

### CI/CD

- Unblock inherited rustc, cargo-deny, and maturin failures ([#42](https://github.com/sophie4869/ragfs/pull/42))

### Changed

- Fix formatting and clippy warnings

### Fixed

- **python**: Correct import names in __init__.py
- **deps**: Update fuser 0.16 and pyo3 0.27 for security fixes
- Resolve CI failures for v0.2.0 release
- Align docs and FUSE help with real behavior

### Miscellaneous

- Improve project configuration
- Unify project descriptions and metadata across packages

