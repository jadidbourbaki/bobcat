//! Cloudflare's Clef joint schema head, which turns a Qwen3.5 backbone's hidden states into
//! decisions.
//!
//! A Clef request holds a state and a schema of typed questions, each with its allowed options.
//! The backbone reads the rendered request in one prefill. The head then reads the backbone's
//! normalized last hidden states and returns one logit per allowed option of every question. It
//! routes evidence from the whole sequence to each option with cross-attention, lets the
//! questions attend to each other and to the sequence, and scores each option against its
//! question. A softmax over each question's logits gives the option probabilities.
//!
//! The head's tensors live in the backbone's GGUF file under `clef.`, as
//! `tools/clef_gguf.py` writes them. The head runs on the CPU in float32, with its weights
//! dequantized once at load.

use std::ops::Range;

use bobcat_gguf::{Gguf, TensorType};
use safetensors::{Dtype, SafeTensors};

use crate::error::Error;
use crate::lfm2::{require_tensor, require_u32, to_usize};
use crate::qwen35::Model;
use crate::scalar;
use crate::storage::Storage;

/// PyTorch's default `LayerNorm` epsilon, which every norm of the head uses.
const LAYER_NORM_EPS: f32 = 1e-5;

/// The kind of a question, which picks its type embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// A proposition with the options true and false.
    Noul,
    /// A choice among named options.
    Choice,
    /// A level on an ordered scale.
    Score,
}

impl QuestionKind {
    /// Return the row of the type embedding the kind reads.
    fn index(self) -> usize {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
}

/// One question of an encoded request: its kind and the token spans of its instructions and of
/// each allowed option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    /// The kind of the question.
    pub kind: QuestionKind,
    /// The tokens of the question's instructions.
    pub span: Range<usize>,
    /// The tokens of each allowed option, in the order of the logits.
    pub options: Vec<Range<usize>>,
}

/// A dense float32 matrix of `n_rows` rows of `n_cols` elements, with an optional bias per row.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    n_rows: usize,
    n_cols: usize,
}

impl Linear {
    /// Return the products of the `x.len() / n_cols` rows of `x` with the matrix, plus the bias.
    ///
    /// The rows of `x` split across the CPU's cores. The memory projections multiply every token
    /// of the request, which took four fifths of a 300-token decision on one core. Each product
    /// starts scoped threads of its own, about 50 times per decision, which costs far less than
    /// the products. A decision runs once per request, outside the decode loop that keeps a
    /// thread pool alive.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.n_cols;
        let mut out = vec![0.0; rows * self.n_rows];
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let rows_per_thread = rows.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (inputs, outputs) in x
                .chunks(rows_per_thread * self.n_cols)
                .zip(out.chunks_mut(rows_per_thread * self.n_rows))
            {
                scope.spawn(move || self.apply_rows(inputs, outputs));
            }
        });
        out
    }

    /// Write the products of the rows of `inputs` with the matrix, plus the bias, to `outputs`.
    fn apply_rows(&self, inputs: &[f32], outputs: &mut [f32]) {
        for (input, output) in inputs
            .chunks_exact(self.n_cols)
            .zip(outputs.chunks_exact_mut(self.n_rows))
        {
            for (row, (value, weights)) in output
                .iter_mut()
                .zip(self.weight.chunks_exact(self.n_cols))
                .enumerate()
            {
                *value = dot(input, weights) + self.bias.as_ref().map_or(0.0, |b| b[row]);
            }
        }
    }
}

/// Return the dot product of `a` and `b` in eight float sums, which the compiler vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0_f32; 8];
    let (a_chunks, a_rest) = a.as_chunks::<8>();
    let (b_chunks, b_rest) = b.as_chunks::<8>();
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..8 {
            sums[lane] += a[lane] * b[lane];
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(&a, &b)| a * b).sum();
    sums.iter().sum::<f32>() + rest
}

/// A layer norm with a weight and a bias.
#[derive(Debug, Clone)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    /// Return each row of `x` normalized to zero mean and unit variance, scaled, and shifted.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let width = self.weight.len();
        let mut out = vec![0.0; x.len()];
        for (input, output) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            let mean = input.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
            let variance = input
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the double statistics round to float scales"
            )]
            let (mean, scale) = (
                mean as f32,
                (1.0 / (variance + f64::from(LAYER_NORM_EPS)).sqrt()) as f32,
            );
            for (((o, &v), &w), &b) in output
                .iter_mut()
                .zip(input)
                .zip(&self.weight)
                .zip(&self.bias)
            {
                *o = (v - mean) * scale * w + b;
            }
        }
        out
    }
}

/// PyTorch's `MultiheadAttention`, whose file stacks the query, key, and value projections in one
/// matrix, then the output projection.
#[derive(Debug, Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out_proj: Linear,
    heads: usize,
}

impl Attention {
    /// Return the attention of each row of `queries` over the rows of `memory`.
    fn apply(&self, queries: &[f32], memory: &[f32]) -> Vec<f32> {
        let width = self.out_proj.n_rows;
        let head_dim = width / self.heads;
        let q = self.q.apply(queries);
        let k = self.k.apply(memory);
        let v = self.v.apply(memory);
        let n_queries = queries.len() / width;
        let n_keys = memory.len() / width;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attended = vec![0.0; n_queries * width];
        let mut scores = vec![0.0; n_keys];
        for query in 0..n_queries {
            for head in 0..self.heads {
                let span = head * head_dim..(head + 1) * head_dim;
                let q = &q[query * width..][span.clone()];
                for (key, score) in scores.iter_mut().enumerate() {
                    *score = scalar::dot(q, &k[key * width..][span.clone()]) * scale;
                }
                scalar::softmax(&mut scores);
                let out = &mut attended[query * width..][span.clone()];
                for (key, &weight) in scores.iter().enumerate() {
                    for (o, &value) in out.iter_mut().zip(&v[key * width..][span.clone()]) {
                        *o += weight * value;
                    }
                }
            }
        }
        self.out_proj.apply(&attended)
    }
}

/// A feed-forward block: a projection, GELU, and a projection back.
#[derive(Debug, Clone)]
struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut hidden = self.up.apply(x);
        for value in &mut hidden {
            *value = gelu(*value);
        }
        self.down.apply(&hidden)
    }
}

/// A layer that routes evidence from the sequence to each option.
#[derive(Debug, Clone)]
struct EvidenceLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: Attention,
    feedforward_norm: LayerNorm,
    feedforward: FeedForward,
}

/// A pre-norm transformer decoder layer over the questions, as PyTorch's
/// `TransformerDecoderLayer` with `norm_first`.
#[derive(Debug, Clone)]
struct DecoderLayer {
    norm1: LayerNorm,
    self_attn: Attention,
    norm2: LayerNorm,
    cross_attn: Attention,
    norm3: LayerNorm,
    feedforward: FeedForward,
}

/// The joint schema head of a Clef model.
#[derive(Debug, Clone)]
pub struct Head {
    hidden_size: usize,
    width: usize,
    hidden_norm: LayerNorm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: Vec<f32>,
    evidence_layers: Vec<EvidenceLayer>,
    option_summary_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    scorer: FeedForward,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
}

/// The shape of a head, as the release's `joint_head_config.json` gives it.
#[derive(Debug, Clone, Copy)]
struct Config {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
}

impl Head {
    /// Load the head from the `clef.` tensors of `model`'s file, or return `None` when the file
    /// holds no head.
    pub fn load(model: &Model) -> Result<Option<Self>, Error> {
        let gguf = &model.gguf;
        if gguf.u32("clef.width").is_none() {
            return Ok(None);
        }
        let key = |key: &str| require_u32(gguf, &format!("clef.{key}")).map(to_usize);
        let config = Config {
            hidden_size: key("hidden_size")?,
            width: key("width")?,
            routing_layers: key("routing_layers")?,
            layers: key("layers")?,
            heads: key("heads")?,
            feedforward: key("feedforward")?,
        };
        Self::build(model, &config, &Loader::Gguf(gguf)).map(Some)
    }

    /// Load the head for `model` from a Clef release's `joint_head_config.json`, whose bytes are
    /// `config`, and its `joint_head.safetensors`, whose bytes are `weights`.
    pub fn from_safetensors(model: &Model, config: &[u8], weights: &[u8]) -> Result<Self, Error> {
        let config: serde_json::Value =
            serde_json::from_slice(config).map_err(|error| Error::Head(error.to_string()))?;
        let key = |key: &str| -> Result<usize, Error> {
            config[key]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::Head(format!("the configuration needs a count {key}")))
        };
        let config = Config {
            hidden_size: key("hidden_size")?,
            width: key("width")?,
            routing_layers: key("routing_layers")?,
            layers: key("layers")?,
            heads: key("heads")?,
            feedforward: key("feedforward")?,
        };
        let tensors =
            SafeTensors::deserialize(weights).map_err(|error| Error::Head(error.to_string()))?;
        Self::build(model, &config, &Loader::Safetensors(&tensors))
    }

    /// Return the head of `config`'s shape, with the tensors of `loader`.
    fn build(model: &Model, config: &Config, loader: &Loader<'_>) -> Result<Self, Error> {
        let Config {
            hidden_size,
            width,
            routing_layers,
            layers,
            heads,
            feedforward,
        } = *config;
        if hidden_size != to_usize(model.hyperparameters().n_embd)
            || width == 0
            || feedforward == 0
            || heads == 0
            || !width.is_multiple_of(heads)
        {
            return Err(Error::Hyperparameters);
        }
        // The configuration comes from the model file, so the multiples of the width it names
        // must fit before any shape uses them.
        let three_widths = width.checked_mul(3).ok_or(Error::Hyperparameters)?;
        let four_widths = width.checked_mul(4).ok_or(Error::Hyperparameters)?;
        let linear = |name: &str, n_cols: usize, n_rows: usize, bias: bool| {
            loader.linear(name, n_cols, n_rows, bias)
        };
        let norm = |name: &str, n: usize| loader.layer_norm(name, n);
        let attention = |name: &str| -> Result<Attention, Error> {
            let weight =
                loader.values(&format!("{name}.in_proj_weight"), &[width, three_widths])?;
            let bias = loader.values(&format!("{name}.in_proj_bias"), &[three_widths])?;
            let part = |i: usize| Linear {
                weight: weight[i * width * width..(i + 1) * width * width].to_vec(),
                bias: Some(bias[i * width..(i + 1) * width].to_vec()),
                n_rows: width,
                n_cols: width,
            };
            Ok(Attention {
                q: part(0),
                k: part(1),
                v: part(2),
                out_proj: linear(&format!("{name}.out_proj"), width, width, true)?,
                heads,
            })
        };
        let feedforward_block = |up: &str, down: &str| -> Result<FeedForward, Error> {
            Ok(FeedForward {
                up: linear(up, width, feedforward, true)?,
                down: linear(down, feedforward, width, true)?,
            })
        };

        let evidence_layers = (0..routing_layers)
            .map(|i| -> Result<EvidenceLayer, Error> {
                let name = |part: &str| format!("evidence_layers.{i}.{part}");
                Ok(EvidenceLayer {
                    query_norm: norm(&name("query_norm"), width)?,
                    memory_norm: norm(&name("memory_norm"), width)?,
                    attention: attention(&name("attention"))?,
                    feedforward_norm: norm(&name("feedforward_norm"), width)?,
                    feedforward: feedforward_block(&name("feedforward.0"), &name("feedforward.3"))?,
                })
            })
            .collect::<Result<_, _>>()?;
        let layers = (0..layers)
            .map(|i| -> Result<DecoderLayer, Error> {
                let name = |part: &str| format!("layers.{i}.{part}");
                Ok(DecoderLayer {
                    norm1: norm(&name("norm1"), width)?,
                    self_attn: attention(&name("self_attn"))?,
                    norm2: norm(&name("norm2"), width)?,
                    cross_attn: attention(&name("multihead_attn"))?,
                    norm3: norm(&name("norm3"), width)?,
                    feedforward: feedforward_block(&name("linear1"), &name("linear2"))?,
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            hidden_size,
            width,
            hidden_norm: norm("hidden_norm", hidden_size)?,
            memory_projection: linear("memory_projection", hidden_size, width, false)?,
            question_projection: linear("question_projection", hidden_size, width, false)?,
            option_question_projection: linear(
                "option_question_projection",
                hidden_size,
                width,
                false,
            )?,
            global_projection: linear("global_projection", hidden_size, width, false)?,
            option_context_projection: linear(
                "option_context_projection",
                hidden_size,
                width,
                false,
            )?,
            option_lexical_projection: linear(
                "option_lexical_projection",
                hidden_size,
                width,
                false,
            )?,
            type_embedding: loader.values("type_embedding.weight", &[width, 3])?,
            evidence_layers,
            option_summary_norm: norm("option_summary_norm", width)?,
            layers,
            field_norm: norm("field_norm", width)?,
            option_norm: norm("option_norm", width)?,
            scorer: FeedForward {
                up: linear("residual_scorer.0", four_widths, width, true)?,
                down: linear("residual_scorer.3", width, 1, true)?,
            },
            prior_logit_scale: loader.scalar("prior_logit_scale")?,
            joint_logit_scale: loader.scalar("joint_logit_scale")?,
            residual_gate: loader.scalar("residual_gate")?,
        })
    }

    /// Return each question's logits, one per allowed option, for the request whose `tokens`
    /// gave the backbone's normalized last hidden states `hidden`, `hidden_size` floats per
    /// token.
    ///
    /// The lexical part of each option reads the rows of the backbone's output matrix for the
    /// option's tokens.
    pub fn logits(
        &self,
        model: &Model,
        hidden: &[f32],
        tokens: &[u32],
        questions: &[Question],
    ) -> Result<Vec<Vec<f32>>, Error> {
        let n_tokens = tokens.len();
        if n_tokens == 0 || hidden.len() != n_tokens * self.hidden_size {
            return Err(Error::Argument(
                "the hidden states must hold hidden_size floats per token",
            ));
        }
        // The options read rows of the output matrix by token.
        let n_vocab = model.hyperparameters().n_vocab;
        if let Some(&token) = tokens.iter().find(|&&token| token >= n_vocab) {
            return Err(Error::Token { token, n_vocab });
        }
        let spans = questions
            .iter()
            .flat_map(|question| std::iter::once(&question.span).chain(&question.options));
        if questions.is_empty()
            || questions.iter().any(|question| question.options.is_empty())
            || spans
                .clone()
                .any(|span| span.start >= span.end || span.end > n_tokens)
        {
            return Err(Error::Argument(
                "every question needs options, and every span must hold tokens of the request",
            ));
        }

        let normalized = self.hidden_norm.apply(hidden);
        let memory = self.memory_projection.apply(&normalized);
        let global = &normalized[(n_tokens - 1) * self.hidden_size..];
        let mean = |span: &Range<usize>| mean_rows(&normalized, self.hidden_size, span);
        let question_vectors: Vec<Vec<f32>> = questions.iter().map(|q| mean(&q.span)).collect();

        // Each option starts from its own tokens' states, the output embeddings of those
        // tokens, and its question.
        let output = &model.output;
        let output_data = &model.gguf.bytes()[output.tensor.data_range()];
        let mut lexical: Vec<Vec<f32>> = Vec::new();
        let mut option_queries = Vec::new();
        for (question, question_vector) in questions.iter().zip(&question_vectors) {
            let from_question = self.option_question_projection.apply(question_vector);
            for span in &question.options {
                let mut embedding = vec![0.0; self.hidden_size];
                let mut row = vec![0.0; self.hidden_size];
                for &token in &tokens[span.clone()] {
                    scalar::get_row(
                        output.tensor.data_type(),
                        output_data,
                        to_usize(token),
                        &mut row,
                    );
                    add_in_place(&mut embedding, &row);
                }
                let count = span.len() as f32;
                for value in &mut embedding {
                    *value /= count;
                }
                let mut query = self.option_context_projection.apply(&mean(span));
                add_in_place(
                    &mut query,
                    &self.option_lexical_projection.apply(&embedding),
                );
                add_in_place(&mut query, &from_question);
                option_queries.extend(query);
                lexical.push(embedding);
            }
        }

        let mut routed = option_queries;
        for layer in &self.evidence_layers {
            let memory = layer.memory_norm.apply(&memory);
            let attended = layer
                .attention
                .apply(&layer.query_norm.apply(&routed), &memory);
            add_in_place(&mut routed, &attended);
            let fed = layer
                .feedforward
                .apply(&layer.feedforward_norm.apply(&routed));
            add_in_place(&mut routed, &fed);
        }

        // Each question's field starts from its instructions, a summary of its options weighted
        // by their agreement with the instructions, the last token's state, and its kind.
        let global_field = self.global_projection.apply(global);
        let mut fields = Vec::new();
        let mut first_option = 0;
        let mut summaries = Vec::new();
        for (question, question_vector) in questions.iter().zip(&question_vectors) {
            let field = self.question_projection.apply(question_vector);
            let options =
                &routed[first_option * self.width..][..question.options.len() * self.width];
            let mut weights: Vec<f32> = options
                .chunks_exact(self.width)
                .map(|option| scalar::dot(option, &field) / (self.width as f32).sqrt())
                .collect();
            scalar::softmax(&mut weights);
            let mut summary = vec![0.0; self.width];
            for (option, &weight) in options.chunks_exact(self.width).zip(&weights) {
                for (s, &value) in summary.iter_mut().zip(option) {
                    *s += weight * value;
                }
            }
            summaries.extend(summary);
            fields.extend(field);
            first_option += question.options.len();
        }
        let summaries = self.option_summary_norm.apply(&summaries);
        add_in_place(&mut fields, &summaries);
        for (field, question) in fields.chunks_exact_mut(self.width).zip(questions) {
            add_in_place(field, &global_field);
            let kind = question.kind.index();
            add_in_place(
                field,
                &self.type_embedding[kind * self.width..(kind + 1) * self.width],
            );
        }

        for layer in &self.layers {
            let normed = layer.norm1.apply(&fields);
            let attended = layer.self_attn.apply(&normed, &normed);
            add_in_place(&mut fields, &attended);
            let attended = layer.cross_attn.apply(&layer.norm2.apply(&fields), &memory);
            add_in_place(&mut fields, &attended);
            let fed = layer.feedforward.apply(&layer.norm3.apply(&fields));
            add_in_place(&mut fields, &fed);
        }
        let fields = self.field_norm.apply(&fields);

        let prior_scale = self.prior_logit_scale.min(100.0_f32.ln()).exp();
        let joint_scale = self.joint_logit_scale.min(100.0_f32.ln()).exp();
        let gate = 1.0 / (1.0 + (-self.residual_gate).exp());
        let mut logits = Vec::with_capacity(questions.len());
        let mut first_option = 0;
        for ((question, question_vector), field) in questions
            .iter()
            .zip(&question_vectors)
            .zip(fields.chunks_exact(self.width))
        {
            let n_options = question.options.len();
            let mut anchor: Vec<f32> = question_vector
                .iter()
                .zip(global)
                .map(|(q, g)| q + g)
                .collect();
            normalize(&mut anchor);
            let options = self
                .option_norm
                .apply(&routed[first_option * self.width..][..n_options * self.width]);
            let mut features = Vec::with_capacity(n_options * 4 * self.width);
            for option in options.chunks_exact(self.width) {
                features.extend_from_slice(field);
                features.extend_from_slice(option);
                features.extend(field.iter().zip(option).map(|(f, o)| f * o));
                features.extend(field.iter().zip(option).map(|(f, o)| (f - o).abs()));
            }
            let residuals = self.scorer.apply(&features);
            let mut question_logits = Vec::with_capacity(n_options);
            for (o, option) in options.chunks_exact(self.width).enumerate() {
                let mut lexical_anchor = lexical[first_option + o].clone();
                normalize(&mut lexical_anchor);
                let prior = prior_scale * scalar::dot(&lexical_anchor, &anchor);
                let joint = joint_scale * cosine(field, option) + residuals[o];
                question_logits.push(prior + gate * joint);
            }
            logits.push(question_logits);
            first_option += n_options;
        }
        Ok(logits)
    }
}

/// Reads the head's tensors out of a GGUF file or a safetensors file.
enum Loader<'s> {
    /// A GGUF file that holds the head's tensors under `clef.` names.
    Gguf(&'s Gguf<Storage>),
    /// The release's `joint_head.safetensors`.
    Safetensors(&'s SafeTensors<'s>),
}

impl Loader<'_> {
    /// Return the float values of the head tensor `NAME`, whose GGUF shape must be `want`. A
    /// safetensors file lists the same dimensions in reverse, and holds each scalar with no
    /// dimensions at all.
    fn values(&self, name: &str, want: &[usize]) -> Result<Vec<f32>, Error> {
        let gguf = match self {
            Self::Gguf(gguf) => gguf,
            Self::Safetensors(tensors) => {
                let view = tensors
                    .tensor(name)
                    .map_err(|_| Error::MissingTensor(name.to_owned()))?;
                let mut shape: Vec<usize> = view.shape().iter().rev().copied().collect();
                if shape.is_empty() {
                    shape.push(1);
                }
                if shape != want {
                    return Err(Error::Head(format!(
                        "{name} has shape {shape:?}, and the head needs {want:?}"
                    )));
                }
                let data_type = match view.dtype() {
                    Dtype::F32 => TensorType::F32,
                    Dtype::F16 => TensorType::F16,
                    Dtype::BF16 => TensorType::Bf16,
                    other => {
                        return Err(Error::Head(format!("{name} holds {other:?} values")));
                    }
                };
                let mut values = vec![0.0; want.iter().product()];
                if view.data().len() != values.len() * data_type.block().1 {
                    return Err(Error::Head(format!(
                        "{name} holds the wrong number of bytes"
                    )));
                }
                scalar::dequantize(data_type, view.data(), &mut values);
                return Ok(values);
            }
        };
        let name = format!("clef.{name}");
        let want: Vec<u64> = want.iter().map(|&n| n as u64).collect();
        let tensor = require_tensor(gguf, &name, &want)?;
        let data_type = tensor.data_type();
        if !matches!(
            data_type,
            TensorType::F32 | TensorType::F16 | TensorType::Bf16
        ) {
            return Err(Error::TensorType { name, data_type });
        }
        let bytes = gguf
            .tensor_data(tensor)
            .ok_or_else(|| Error::MissingTensor(name.clone()))?;
        let count = want.iter().product::<u64>();
        let mut values = vec![0.0; usize::try_from(count).map_err(|_| Error::Hyperparameters)?];
        scalar::dequantize(data_type, bytes, &mut values);
        Ok(values)
    }

    /// Return the matrix `NAME.weight` of `n_rows` rows of `n_cols` elements, with the bias
    /// `NAME.bias` when `bias` holds.
    fn linear(
        &self,
        name: &str,
        n_cols: usize,
        n_rows: usize,
        bias: bool,
    ) -> Result<Linear, Error> {
        Ok(Linear {
            weight: self.values(&format!("{name}.weight"), &[n_cols, n_rows])?,
            bias: if bias {
                Some(self.values(&format!("{name}.bias"), &[n_rows])?)
            } else {
                None
            },
            n_rows,
            n_cols,
        })
    }

    fn layer_norm(&self, name: &str, n: usize) -> Result<LayerNorm, Error> {
        Ok(LayerNorm {
            weight: self.values(&format!("{name}.weight"), &[n])?,
            bias: self.values(&format!("{name}.bias"), &[n])?,
        })
    }

    /// Return the single value of the head tensor `NAME`.
    fn scalar(&self, name: &str) -> Result<f32, Error> {
        Ok(self.values(name, &[1])?[0])
    }
}

/// Return the mean of the rows of `x`, `width` floats each, that `span` covers.
fn mean_rows(x: &[f32], width: usize, span: &Range<usize>) -> Vec<f32> {
    let mut mean = vec![0.0; width];
    for row in x[span.start * width..span.end * width].chunks_exact(width) {
        add_in_place(&mut mean, row);
    }
    let count = span.len() as f32;
    for value in &mut mean {
        *value /= count;
    }
    mean
}

/// Add `delta` to `x` element by element.
fn add_in_place(x: &mut [f32], delta: &[f32]) {
    for (x, &d) in x.iter_mut().zip(delta) {
        *x += d;
    }
}

/// Divide `x` by its L2 norm, or by 1e-12 when the norm is smaller, as PyTorch's `normalize`
/// does.
fn normalize(x: &mut [f32]) {
    let norm = scalar::dot(x, x).sqrt().max(1e-12);
    for value in x {
        *value /= norm;
    }
}

/// Return the cosine similarity of `a` and `b`, guarded by 1e-8 as PyTorch's
/// `cosine_similarity` is.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let norms = (scalar::dot(a, a) * scalar::dot(b, b)).sqrt();
    scalar::dot(a, b) / norms.max(1e-8)
}

/// Return the exact GELU of `x`, `x * Φ(x)` with the standard normal CDF `Φ`.
fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the double result rounds to the float activation"
    )]
    let value = (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32;
    value
}

/// Return the error function of `x`, with an absolute error below 1.2e-7.
///
/// The approximation is Numerical Recipes' `erfcc`, a Chebyshev fit of the complementary error
/// function.
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}
