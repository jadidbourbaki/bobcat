//! The choice of each next token from the model's logits.
//!
//! The steps run in llama.cpp's order: the repetition penalty, top-k, top-p, min-p, the
//! temperature, then a random draw. A temperature of zero picks the most likely token.

use gip_gguf::Gguf;

/// How the sampler chooses tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Settings {
    /// The softmax temperature. Zero picks the most likely token.
    pub(crate) temperature: f32,
    /// The number of most likely tokens kept. Zero keeps every token.
    pub(crate) top_k: usize,
    /// The smallest set of tokens whose probabilities sum to at least this much is kept.
    pub(crate) top_p: f32,
    /// Tokens less likely than this fraction of the most likely token are dropped.
    pub(crate) min_p: f32,
    /// The logit divisor for tokens among the last `repeat_last_n`. One turns the penalty off.
    pub(crate) repeat_penalty: f32,
    /// The number of recent tokens the repetition penalty covers.
    pub(crate) repeat_last_n: usize,
}

impl Settings {
    /// llama.cpp's defaults, for settings the model file leaves out.
    const FALLBACK: Self = Self {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
    };

    /// Return the sampling settings the metadata of `gguf` recommends, with llama.cpp's defaults
    /// for the rest.
    pub(crate) fn recommended(gguf: &Gguf<impl AsRef<[u8]>>) -> Self {
        let fallback = Self::FALLBACK;
        Self {
            temperature: gguf
                .f32("general.sampling.temp")
                .unwrap_or(fallback.temperature),
            top_k: gguf
                .u32("general.sampling.top_k")
                .map_or(fallback.top_k, |k| k as usize),
            top_p: gguf.f32("general.sampling.top_p").unwrap_or(fallback.top_p),
            min_p: gguf.f32("general.sampling.min_p").unwrap_or(fallback.min_p),
            repeat_penalty: gguf
                .f32("general.sampling.penalty_repeat")
                .unwrap_or(fallback.repeat_penalty),
            repeat_last_n: gguf
                .u32("general.sampling.penalty_last_n")
                .map_or(fallback.repeat_last_n, |n| n as usize),
        }
    }

    /// Report whether the settings always pick the most likely token, which the GPU can do
    /// itself.
    pub(crate) fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 && !self.penalizes()
    }

    /// Report whether the repetition penalty changes any logit.
    fn penalizes(&self) -> bool {
        (self.repeat_penalty - 1.0).abs() > f32::EPSILON
    }
}

/// Chooses tokens from logits.
pub(crate) struct Sampler {
    settings: Settings,
    rng: fastrand::Rng,
    /// Candidate token ids and their logits, then probabilities, reused across tokens.
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    /// Return a sampler with `settings` whose random draws start from `seed`.
    pub(crate) fn new(settings: Settings, seed: u64) -> Self {
        Self {
            settings,
            rng: fastrand::Rng::with_seed(seed),
            candidates: Vec::new(),
        }
    }

    /// Report whether the sampler always picks the most likely token.
    pub(crate) fn is_greedy(&self) -> bool {
        self.settings.is_greedy()
    }

    /// Choose the next token from `logits`, which the penalty for the tokens in `recent` changes
    /// in place.
    pub(crate) fn sample(&mut self, logits: &mut [f32], recent: &[u32]) -> u32 {
        let settings = self.settings;
        if settings.penalizes() {
            let start = recent.len().saturating_sub(settings.repeat_last_n);
            for &token in &recent[start..] {
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

        if settings.top_k > 0 && settings.top_k < candidates.len() {
            candidates.select_nth_unstable_by(settings.top_k, by_logit);
            candidates.truncate(settings.top_k);
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

    fn settings(temperature: f32) -> Settings {
        Settings {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
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
