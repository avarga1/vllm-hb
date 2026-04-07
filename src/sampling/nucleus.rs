//! Temperature scaling + top-p nucleus filtering.
//!
//! All functions are pure over a `&mut [f32]` probability slice.

/// Divide every logit by `temperature` (equivalent to sharpening the
/// distribution for temperature < 1.0, flattening it for > 1.0).
pub fn apply_temperature(probs: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for p in probs.iter_mut() {
        *p *= inv_temp;
    }
}

/// Numerically stable softmax in-place.
pub fn softmax_inplace(probs: &mut [f32]) {
    let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for p in probs.iter_mut() {
        *p = (*p - max).exp();
        sum += *p;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
}

/// Zero out tokens outside the top-p nucleus.
///
/// Sorts token indices by descending probability, accumulates until the
/// cumulative mass ≥ `top_p`, then zeros everything outside that set.
pub fn nucleus_filter(probs: &mut [f32], top_p: f32) {
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0_f32;
    let mut cutoff = probs.len();

    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff = rank + 1;
            break;
        }
    }

    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }
}

/// Re-normalise a probability slice that may have been zeroed by
/// nucleus filtering.
pub fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn nucleus_filter_keeps_top_mass() {
        let mut v = vec![0.6_f32, 0.3, 0.05, 0.05];
        nucleus_filter(&mut v, 0.9);
        assert_eq!(v[2], 0.0);
        assert_eq!(v[3], 0.0);
        assert!(v[0] > 0.0);
        assert!(v[1] > 0.0);
    }
}
