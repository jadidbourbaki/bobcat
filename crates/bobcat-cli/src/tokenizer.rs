//! The tokenizer a GGUF file describes, rebuilt as a Hugging Face tokenizer.
//!
//! A GGUF file stores its tokenizer's vocabulary, merges, and token types, and names its
//! pre-tokenizer in `tokenizer.ggml.pre`. The regex behind each name lives in the engine, as in
//! llama.cpp and mistral.rs. bobcat knows only the names below and refuses any other, because a
//! wrong regex tokenizes text silently wrong.

use bobcat_gguf::Gguf;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::Error;

/// The pre-tokenizer regex of Llama 3, which LFM2 shares. The pattern matches the one in LFM2's
/// `tokenizer.json`.
const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The pre-tokenizer regex of Qwen2, which splits digits one at a time. Cloudflare's Clef releases
/// tokenize with it. The pattern matches the one in their `tokenizer.json`.
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The pre-tokenizer regex of Qwen3.5, which also keeps combining marks in words. The pattern
/// matches the one in Qwen3.5's `tokenizer.json`.
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The GGUF token type of control tokens such as `<|im_start|>`.
const CONTROL_TOKEN: u32 = 3;

/// The GGUF token type of tokens a model's authors added to the vocabulary, such as `<think>`.
const USER_DEFINED_TOKEN: u32 = 4;

/// The GGUF token type of placeholders that pad the vocabulary.
const UNUSED_TOKEN: u32 = 5;

/// Return the tokenizer that the metadata of `gguf` describes.
pub(crate) fn from_gguf(gguf: &Gguf<impl AsRef<[u8]>>) -> Result<Tokenizer, Error> {
    let model = gguf.string("tokenizer.ggml.model").unwrap_or_default();
    if model != b"gpt2" {
        return Err(format!(
            "bobcat cannot rebuild a {} tokenizer from the model file, so pass the model's \
             tokenizer.json with --tokenizer",
            String::from_utf8_lossy(model)
        )
        .into());
    }
    let pre = gguf.string("tokenizer.ggml.pre").unwrap_or_default();
    // llama.cpp ignores merges for LFM2, taking any whole word the vocabulary holds as one token.
    // Qwen models merge as their tokenizer.json does, after NFC normalization.
    let (pattern, ignore_merges, nfc) = match pre {
        b"lfm2" => (LLAMA3_PATTERN, true, false),
        b"qwen2" => (QWEN2_PATTERN, false, true),
        b"qwen35" => (QWEN35_PATTERN, false, true),
        other => {
            return Err(format!(
                "bobcat cannot rebuild the {} pre-tokenizer from the model file, so pass the \
                 model's tokenizer.json with --tokenizer",
                String::from_utf8_lossy(other)
            )
            .into());
        }
    };

    // Each token keeps its id and GGUF token type. Converters pad the vocabulary to the size of
    // the embedding matrix with unused placeholders, which no text maps to.
    let tokens = (0_u32..)
        .zip(
            gguf.string_array("tokenizer.ggml.tokens")
                .ok_or("the model file holds no tokenizer vocabulary")?,
        )
        .map(|(id, token)| {
            let token_type = gguf
                .array_u32("tokenizer.ggml.token_type", u64::from(id))
                .ok_or("the model file holds no tokenizer token types")?;
            Ok((id, String::from_utf8(token.to_vec())?, token_type))
        })
        .filter(|token| !matches!(token, Ok((_, _, UNUSED_TOKEN))))
        .collect::<Result<Vec<_>, Error>>()?;
    let merges = gguf
        .string_array("tokenizer.ggml.merges")
        .ok_or("the model file holds no tokenizer merges")?
        .map(|merge| {
            let merge = std::str::from_utf8(merge)?;
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| format!("the tokenizer merge {merge:?} has no space"))?;
            Ok((left.to_owned(), right.to_owned()))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let vocab: Vocab = tokens
        .iter()
        .map(|(id, token, _)| (token.clone(), *id))
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(ignore_merges)
        .build()?;
    let mut tokenizer = Tokenizer::new(bpe);
    if nfc {
        tokenizer.with_normalizer(Some(NFC))?;
    }
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        Split::new(
            SplitPattern::Regex(pattern.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )?
        .into(),
        ByteLevel::new(false, true, false).into(),
    ])));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    // Control and user-defined tokens match as whole units wherever they appear in text.
    let added = tokens
        .iter()
        .filter(|(_, _, token_type)| matches!(*token_type, CONTROL_TOKEN | USER_DEFINED_TOKEN))
        .map(|(_, token, token_type)| {
            AddedToken::from(token.clone(), *token_type == CONTROL_TOKEN)
        });
    tokenizer.add_tokens(added.collect::<Vec<_>>())?;
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// Text that exercises every branch of the pre-tokenizer regex, the scripts LFM2.5 supports,
    /// and the control and user-defined tokens chat templates write.
    const SAMPLES: &[&str] = &[
        "The quick brown fox jumps over the lazy dog.",
        "I'm sure they'll say it's what we'd've done, but you're wrong.",
        "Pi is 3.14159, and 1234567 people live in 2026.",
        "fn main() {\n    println!(\"hi\");\n}\n\n\n\tindented\r\n  two  spaces   ",
        "巴黎是法国的首都。东京是日本的首都。",
        "日本語のテキストとカタカナ、ひらがな。",
        "مرحبا بالعالم، كيف حالك؟",
        "नमस्ते दुनिया, आप कैसे हैं?",
        "안녕하세요 세계, 잘 지내세요?",
        "Привет, мир! Как дела?",
        "Emoji 😀🎉 and symbols ∑∫√ ©®™",
        "<|startoftext|><|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>",
        "<think>Let me count.</think>The answer is 4.<|im_end|>",
    ];

    /// Check that the tokenizer rebuilt from each model file matches the model's own
    /// `tokenizer.json` on the whole vocabulary and on every sample.
    #[test]
    fn gguf_tokenizer_matches_tokenizer_json() -> Result<(), Error> {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let lfm2 = ["LFM2.5-350M", "LFM2.5-1.2B-Instruct", "LFM2.5-2.6B"].map(|name| {
            (
                format!("{name}-Q8_0.gguf"),
                format!("{name}-MLX-8bit/tokenizer.json"),
            )
        });
        let qwen35 = [
            ("Qwen3.5-0.8B-Q8_0.gguf", "hf/Qwen3.5-0.8B/tokenizer.json"),
            ("clef-flash-Q8_0.gguf", "hf/clef-flash/tokenizer.json"),
        ]
        .map(|(gguf, json)| (gguf.to_owned(), json.to_owned()));
        for (name, json) in lfm2.into_iter().chain(qwen35) {
            let gguf_path = models.join(&name);
            let json_path = models.join(json);
            if !gguf_path.exists() || !json_path.exists() {
                eprintln!(
                    "skip: needs {} and {}",
                    gguf_path.display(),
                    json_path.display()
                );
                continue;
            }
            let storage = std::fs::read(&gguf_path)?;
            let gguf = Gguf::parse(storage)?;
            let rebuilt = from_gguf(&gguf)?;
            let reference = Tokenizer::from_file(&json_path)?;

            // llama.cpp's converter appends Qwen3.5's audio control tokens after the
            // vocabulary of its tokenizer.json, and no text maps to them.
            let vocab_size = reference.get_vocab_size(true);
            assert!(
                rebuilt.get_vocab_size(true) >= vocab_size,
                "{name} vocabulary size"
            );
            for id in 0..u32::try_from(vocab_size)? {
                assert_eq!(
                    rebuilt.id_to_token(id),
                    reference.id_to_token(id),
                    "{name} token {id}"
                );
            }
            for sample in SAMPLES {
                let got = rebuilt.encode(*sample, false)?;
                let want = reference.encode(*sample, false)?;
                assert_eq!(got.get_ids(), want.get_ids(), "{name} encodes {sample:?}");
                assert_eq!(
                    rebuilt.decode(got.get_ids(), false)?,
                    reference.decode(want.get_ids(), false)?,
                    "{name} decodes {sample:?}"
                );
            }
        }
        Ok(())
    }
}
