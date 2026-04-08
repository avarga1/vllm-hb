//! Stop-sequence matching.
//!
//! Checks whether the text generated so far ends with any of the caller-
//! supplied stop strings and signals the worker to halt generation early.
//!
//! # Stub status
//! Types and interface are final; matching logic is `todo!()`.
//! See issue #21.

/// Checks a rolling output buffer against a list of stop strings.
///
/// Call [`StopChecker::check`] after every new token is appended to the
/// output.  When a match is found the checker returns `true` and the caller
/// should stop generation and trim the matched suffix from the output.
pub struct StopChecker {
    /// Stop strings supplied by the client, pre-validated to be non-empty.
    _stop: Vec<String>,
    /// Rolling suffix buffer — kept to `max_stop_len` bytes so we don't scan
    /// the entire output on every token.
    _buf: String,
}

impl StopChecker {
    /// Create a new checker for the given stop strings.
    ///
    /// Returns `None` when `stop` is empty (fast path: no checking needed).
    pub fn new(stop: Vec<String>) -> Option<Self> {
        if stop.is_empty() {
            return None;
        }
        Some(Self {
            _stop: stop,
            _buf: String::new(),
        })
    }

    /// Append `token_text` to the rolling buffer and test for any stop match.
    ///
    /// Returns `true` if generation should halt.  The caller is responsible
    /// for trimming the matched stop string from the final output.
    pub fn check(&mut self, _token_text: &str) -> bool {
        todo!("issue #21 — stop sequence matching")
    }

    /// Returns the stop string that matched, if any was found.
    ///
    /// Only meaningful after [`check`] returns `true`.
    pub fn matched(&self) -> Option<&str> {
        todo!("issue #21 — stop sequence matching")
    }
}
