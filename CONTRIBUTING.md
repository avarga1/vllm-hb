# Contributing to vllm-hb

Thanks for taking the time. This is a small, focused project — PRs welcome.

## Before you open a PR

- **Check existing issues** — someone may already be working on it.
- **Open an issue first** for anything non-trivial (new model support, scheduler
  changes, API surface changes). Saves everyone time before code is written.
- Small fixes (typos, docs, obvious bugs) — just open the PR directly.

## Setup

You need Rust stable (1.75+). CUDA is optional for most work.

```bash
git clone https://github.com/avarga1/vllm-hb
cd vllm-hb

# CPU build — works on any machine, no CUDA required
cargo build-cpu

# Run unit tests (no GPU needed)
cargo test-cpu

# GPU build — requires CUDA 12.x + nvcc in PATH
cargo build-gpu
```

Aliases are defined in [`.cargo/config.toml`](.cargo/config.toml).

## What we want

| Area | Status | Notes |
|---|---|---|
| Bug fixes | Always welcome | Include a test that reproduces the bug |
| New model support | Welcome | Prefer `candle-transformers` models; open issue first |
| Continuous batching / paged attention | Wanted | Big change — discuss in an issue first |
| Multi-GPU (tensor parallel) | Wanted | Big change — discuss in an issue first |
| OpenAI API completeness | Welcome | `/v1/completions`, function calling, logprobs |
| Performance improvements | Welcome | Include benchmark numbers before/after |
| Docs / README | Always welcome | — |

## Code style

- `cargo fmt` before committing. CI enforces this.
- `cargo clippy -- -D warnings` must pass. CI enforces this.
- No `unwrap()` in non-test code. Use `?` or explicit error handling.
- Keep `unsafe` minimal and document every block.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Mistral model support
fix: correct nucleus filter boundary condition
perf: avoid allocation in sampling hot path
docs: add multi-GPU roadmap to README
ci: cache Cargo registry between runs
```

## Pull request checklist

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test --no-default-features` passes
- [ ] New behaviour has a test (unit or integration)
- [ ] CHANGELOG.md updated under `[Unreleased]`

## Reporting bugs

Open a GitHub issue with:
- vllm-hb version / commit SHA
- GPU model + CUDA version
- Model you were running
- Command you ran
- Full error output / stack trace

## License

By contributing you agree your contributions are licensed under Apache-2.0,
the same license as this project.
