//! The choice of each next token from the model's logits.
//!
//! The steps run in llama.cpp's order: the repetition penalty, top-k, top-p, min-p, the
//! temperature, then a random draw. A temperature of zero picks the most likely token.

use gip::Sampling;

/// The number of recent tokens the repetition penalty covers, as in llama.cpp.
const REPEAT_LAST_N: usize = 64;

/// Report whether the repetition penalty of `settings` changes any logit.
fn penalizes(settings: &Sampling) -> bool {
    (settings.repeat_penalty - 1.0).abs() > f32::EPSILON
}

/// Chooses tokens from logits.
pub(crate) struct Sampler {
    settings: Sampling,
    rng: fastrand::Rng,
    /// Candidate token ids and their logits, then probabilities, reused across tokens.
    candidates: Vec<(u32, f32)>,
    /// The distinct tokens the repetition penalty covers, reused across tokens.
    penalized: Vec<u32>,
}

impl Sampler {
    /// Return a sampler with `settings` whose random draws start from `seed`.
    pub(crate) fn new(settings: Sampling, seed: u64) -> Self {
        Self {
            settings,
            rng: fastrand::Rng::with_seed(seed),
            candidates: Vec::new(),
            penalized: Vec::new(),
        }
    }

    /// Report whether the sampler always picks the most likely token, which the GPU can do
    /// itself.
    pub(crate) fn is_greedy(&self) -> bool {
        self.settings.temperature <= 0.0 && !penalizes(&self.settings)
    }

    /// Choose the next token from `logits`, which the penalty for the tokens in `recent` changes
    /// in place.
    pub(crate) fn sample(&mut self, logits: &mut [f32], recent: &[u32]) -> u32 {
        let settings = self.settings;
        if penalizes(&settings) {
            // Each distinct recent token takes the penalty once, however often it repeats, as in
            // llama.cpp and transformers.
            let start = recent.len().saturating_sub(REPEAT_LAST_N);
            self.penalized.clear();
            self.penalized.extend_from_slice(&recent[start..]);
            self.penalized.sort_unstable();
            self.penalized.dedup();
            for &token in &self.penalized {
                if let Some(logit) = logits.get_mut(token as usize) {
                    *logit = if *logit > 0.0 {
                        *logit / settings.repeat_penalty
                    } else {
                        *logit * settings.repeat_penalty
                    };
                }
            }
        }

        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.extend((0_u32..).zip(logits.iter().copied()));
        let by_logit = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if settings.temperature <= 0.0 {
            return candidates
                .iter()
                .copied()
                .min_by(by_logit)
                .map_or(0, |c| c.0);
        }

        let top_k = settings.top_k as usize;
        if top_k > 0 && top_k < candidates.len() {
            candidates.select_nth_unstable_by(top_k, by_logit);
            candidates.truncate(top_k);
        }
        candidates.sort_unstable_by(by_logit);

        // Softmax at temperature one ranks the candidates for top-p and min-p, as in llama.cpp.
        softmax(candidates, 1.0);
        let mut cumulative = 0.0;
        let keep = candidates
            .iter()
            .position(|&(_, p)| {
                cumulative += p;
                cumulative >= settings.top_p
            })
            .map_or(candidates.len(), |last| last + 1);
        candidates.truncate(keep);
        let floor = candidates.first().map_or(0.0, |c| c.1) * settings.min_p;
        candidates.retain(|&(_, p)| p >= floor);

        // The temperature applies to the surviving logits, which the probabilities encode.
        for candidate in candidates.iter_mut() {
            candidate.1 = candidate.1.ln();
        }
        softmax(candidates, settings.temperature);
        let mut draw = self.rng.f32();
        for &(token, probability) in candidates.iter() {
            if draw < probability {
                return token;
            }
            draw -= probability;
        }
        candidates.last().map_or(0, |c| c.0)
    }
}

/// Replace the logits in `candidates` with their softmax at `temperature`.
fn softmax(candidates: &mut [(u32, f32)], temperature: f32) {
    let max = candidates
        .iter()
        .map(|c| c.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for candidate in candidates.iter_mut() {
        candidate.1 = ((candidate.1 - max) / temperature).exp();
        sum += candidate.1;
    }
    for candidate in candidates.iter_mut() {
        candidate.1 /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(temperature: f32) -> Sampling {
        Sampling {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
        }
    }

    #[test]
    fn zero_temperature_picks_the_largest_logit() {
        let mut sampler = Sampler::new(settings(0.0), 1);
        assert_eq!(sampler.sample(&mut [0.1, 3.0, 2.0], &[]), 1);
    }

    #[test]
    fn penalty_moves_the_choice_off_recent_tokens() {
        let mut penalized = settings(0.0);
        penalized.repeat_penalty = 2.0;
        let mut sampler = Sampler::new(penalized, 1);
        assert_eq!(sampler.sample(&mut [0.1, 3.0, 2.0], &[1]), 2);
        // A token that repeats takes the penalty once, so 3.0 falls to 1.5 and still leads.
        assert_eq!(sampler.sample(&mut [0.1, 3.0, 1.0], &[1, 1]), 1);
    }

    #[test]
    fn top_k_and_min_p_bound_the_draws() {
        let mut bounded = settings(1.0);
        bounded.top_k = 2;
        let mut sampler = Sampler::new(bounded, 7);
        for _ in 0..200 {
            let token = sampler.sample(&mut [5.0, 4.9, -3.0, 4.8], &[]);
            assert!(token == 0 || token == 1, "top-k kept token {token}");
        }
        let mut floored = settings(1.0);
        floored.min_p = 0.5;
        let mut sampler = Sampler::new(floored, 7);
        for _ in 0..200 {
            assert_eq!(sampler.sample(&mut [5.0, 0.0, 0.0], &[]), 0);
        }
    }
}
