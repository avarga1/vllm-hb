//! Stop-sequence matching.
//!
//! Checks whether the rolling output text ends with any of the caller-supplied
//! stop strings and signals the worker to halt generation early.
//!
//! # Design
//!
//! After every token we append its decoded text to a rolling suffix buffer.
//! The buffer is kept at most `max_stop_len` bytes — the length of the longest
//! stop string — so the scan cost per token is O(|stop_strings| × max_len)
//! and we never scan the full output.
//!
//! The caller trims the matched suffix from the final output before sending
//! it to the client.

/// Checks a rolling output buffer against a list of stop strings.
///
/// Create one per sequence with [`StopChecker::new`].  After every token, call
/// [`StopChecker::push`] which returns `true` when generation should stop.
/// Use [`StopChecker::strip_match`] to trim the matched suffix from the final
/// assembled output.
pub struct StopChecker {
    /// Non-empty stop strings supplied by the client.
    stop: Vec<String>,
    /// Length of the longest stop string, in bytes.  The buffer never grows
    /// beyond this so suffix scanning stays O(1) in output length.
    max_len: usize,
    /// Rolling suffix of decoded output text.
    buf: String,
    /// The stop string that last matched, if any.
    matched: Option<String>,
}

impl StopChecker {
    /// Create a new checker for the given stop strings.
    ///
    /// Returns `None` when `stop` is empty — the fast path where no checking
    /// is ever needed.
    pub fn new(stop: Vec<String>) -> Option<Self> {
        let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();
        if stop.is_empty() {
            return None;
        }
        let max_len = stop.iter().map(|s| s.len()).max().unwrap_or(0);
        Some(Self {
            stop,
            max_len,
            buf: String::with_capacity(max_len * 2),
            matched: None,
        })
    }

    /// Append `token_text` to the rolling buffer and test for a stop match.
    ///
    /// Returns `true` if generation should halt.  After a `true` return,
    /// call [`StopChecker::strip_match`] to remove the stop suffix from the
    /// assembled output string.
    pub fn push(&mut self, token_text: &str) -> bool {
        self.buf.push_str(token_text);

        // Trim the head of the buffer so it stays at most `max_len` bytes.
        // We need to trim on a char boundary.
        if self.buf.len() > self.max_len {
            let excess = self.buf.len() - self.max_len;
            // Advance past the first `excess` bytes, staying on a char boundary.
            let trim_at = self
                .buf
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i < excess)
                .last()
                .map(|i| {
                    // Find the start of the *next* char after this byte offset.
                    self.buf[i..]
                        .char_indices()
                        .nth(1)
                        .map(|(j, _)| i + j)
                        .unwrap_or(self.buf.len())
                })
                .unwrap_or(0);
            self.buf.drain(..trim_at);
        }

        // Scan the buffer for any stop string.
        for s in &self.stop {
            if self.buf.ends_with(s.as_str()) {
                self.matched = Some(s.clone());
                return true;
            }
        }
        false
    }

    /// Remove the matched stop suffix from `output` in-place.
    ///
    /// Call this once after [`StopChecker::push`] returns `true`, passing the
    /// full assembled output string.  No-op if no match was recorded.
    pub fn strip_match(&self, output: &mut String) {
        let Some(m) = &self.matched else { return };
        if output.ends_with(m.as_str()) {
            output.truncate(output.len() - m.len());
        }
    }

    /// The stop string that matched, if any.
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stop_list_returns_none() {
        assert!(StopChecker::new(vec![]).is_none());
    }

    #[test]
    fn blank_strings_filtered_out() {
        assert!(StopChecker::new(vec!["".into()]).is_none());
    }

    #[test]
    fn no_match_returns_false() {
        let mut c = StopChecker::new(vec!["END".into()]).unwrap();
        assert!(!c.push("hello"));
        assert!(!c.push(" world"));
    }

    #[test]
    fn exact_match_returns_true() {
        let mut c = StopChecker::new(vec!["END".into()]).unwrap();
        assert!(!c.push("hello "));
        assert!(c.push("END"));
        assert_eq!(c.matched(), Some("END"));
    }

    #[test]
    fn match_split_across_tokens() {
        let mut c = StopChecker::new(vec!["</s>".into()]).unwrap();
        assert!(!c.push("<"));
        assert!(!c.push("/"));
        assert!(!c.push("s"));
        assert!(c.push(">"));
        assert_eq!(c.matched(), Some("</s>"));
    }

    #[test]
    fn strip_match_removes_suffix() {
        let mut c = StopChecker::new(vec!["STOP".into()]).unwrap();
        c.push("helloSTOP");
        let mut out = "helloSTOP".to_string();
        c.strip_match(&mut out);
        assert_eq!(out, "hello");
    }

    #[test]
    fn buffer_trimmed_to_max_stop_len() {
        // stop string is 3 chars; buffer must not grow unboundedly
        let mut c = StopChecker::new(vec!["END".into()]).unwrap();
        for _ in 0..100 {
            c.push("ab");
        }
        // buffer should be at most max_len bytes
        assert!(c.buf.len() <= 3);
    }

    #[test]
    fn first_of_multiple_stop_strings_wins() {
        let mut c = StopChecker::new(vec!["AAA".into(), "BB".into()]).unwrap();
        assert!(c.push("BB"));
        assert_eq!(c.matched(), Some("BB"));
    }
}
