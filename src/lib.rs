//! vllm-hb — Hammingbound inference runtime, library crate.
//!
//! Exposes all internal modules so that the `tests/` integration suite and
//! external tooling (benchmarks, embeddings, etc.) can import them without
//! going through the binary entry point.
//!
//! The binary (`src/main.rs`) is a thin CLI wrapper that delegates entirely
//! to the public API here.

// Working modules
pub mod bench;
pub mod engine;
pub mod sampling;
pub mod server;
pub mod tokenize;
pub mod tools;
pub mod types;
pub mod worker;

// Roadmap stubs (declared so `cargo check` covers them)
pub mod attention;
pub mod parallel;
pub mod scheduler;
pub mod speculative;
