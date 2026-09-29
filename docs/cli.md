# The bobcat command

`crates/bobcat-cli` builds the `bobcat` program. The command calls the `bobcat`
library for everything that touches the model and adds what a person
at a terminal needs: model names, downloads, the tokenizer, the chat
template, and sampling.

## Subcommands

| Command | Job |
|---|---|
| `bobcat respond -m MODEL PROMPT` | Answer one prompt and write the answer to standard output |
| `bobcat chat -m MODEL` | Hold a conversation at the terminal until Ctrl-D |
| `bobcat pull NAME` | Download a model and print the path of its file |
| `bobcat list` | List the downloaded models and their sizes |
| `bobcat rm NAME` | Delete a downloaded model |

The commands follow Unix conventions so they compose with other tools.
Apple's `fm`, Simon Willison's `llm`, and Ollama's `ollama run` served
as references.

- Standard output carries only the answer. `bobcat respond` writes the
  model's thinking to standard error when `--think` is given.
- Text on standard input joins the prompt, so
  `cat notes.md | bobcat respond -m lfm2.5:1.2b "Summarize this."` answers
  about the file.
- `bobcat chat` dims the thinking only when standard output is a terminal
  and `NO_COLOR` is unset.
- A failure prints `bobcat: message` to standard error and exits with
  status 1.

`crates/bobcat-cli/src/main.rs` defines the options with clap and wires
the subcommands together.

## Model names

`-m` and `bobcat pull` take a model in one of three forms.

| Form | Example | Meaning |
|---|---|---|
| Alias | `lfm2.5:1.2b` | A model bobcat supports, in its default quantization |
| Alias with a tag | `lfm2.5:1.2b-q8_0` | The same model in another quantization |
| Hugging Face name | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0` | Any GGUF repository, with an optional tag |

The `ALIASES` table in `crates/bobcat-cli/src/models.rs` maps each alias
to its repository. Aliases default to Q4_K_M, which is half the size of
Q8_0 and decodes faster. Hugging Face names default to Q8_0. A tag
picks the file whose name ends in `-TAG.gguf`. When several files end
in the same tag, the shortest name wins, so `Q4_0` picks
`LFM2.5-350M-Q4_0.gguf` over `LFM2.5-350M-QAD-Q4_0.gguf`. The tag
`QAD-Q4_0` still reaches the longer name.

`-m` also takes a path to a GGUF file.

## Downloads and the cache

bobcat keeps models in the shared Hugging Face cache at
`~/.cache/huggingface/hub`. Python's `huggingface_hub`, transformers,
and other tools read the same cache, so a model downloads once for all
of them. The `hf-hub` crate lists repositories, downloads files, and
scans the cache.

`models::resolve` turns `-m` into a file path. A name already in the
cache resolves at once. A name not yet in the cache downloads first,
so `bobcat chat -m lfm2.5:2.6b` works on a fresh machine. `bobcat pull` shows
a progress line on standard error when standard error is a terminal.

`bobcat rm` deletes the cache entry. The Hugging Face cache stores each
file's data once as a blob and points to the blob from each revision.
`bobcat rm` deletes the blob only when no other cached file points to it.

## The tokenizer

A tokenizer turns text into token ids and back. LFM2 uses byte-level
byte-pair encoding, as GPT-2 and Llama 3 do. A byte-pair encoding
tokenizer first splits text into words with a regex called the
pre-tokenizer, then maps each word to vocabulary tokens.

A GGUF file stores the vocabulary, the merges, and each token's type.
The file names the pre-tokenizer in `tokenizer.ggml.pre` without
storing its regex. `crates/bobcat-cli/src/tokenizer.rs` rebuilds a Hugging
Face `tokenizers` tokenizer from that metadata, as llama.cpp and
mistral.rs do. bobcat knows the regex for the name `lfm2`, which is Llama
3's. bobcat refuses any other name, because a wrong regex tokenizes text
wrongly without any error. `--tokenizer` loads a `tokenizer.json` in
place of the rebuilt tokenizer.

Control tokens such as `<|im_start|>` and tokens such as `<think>`
match as whole units wherever they appear in text. The test
`gguf_tokenizer_matches_tokenizer_json` checks that the rebuilt
tokenizer matches each model's own `tokenizer.json` on the whole
vocabulary and on samples in many scripts.

## The conversation

`crates/bobcat-cli/src/conversation.rs` turns messages into a prompt and
streams the reply.

The model's chat template, stored in `tokenizer.chat_template`, lays
out the messages with the model's special tokens. Chat templates are
Jinja2 programs written for Python. bobcat renders them with MiniJinja.
minijinja-contrib's `pycompat` adds the Python string and dictionary
methods that templates call, such as `.strip()` and `.items()`.

Each reply renders the whole conversation and encodes it. The GPU state
keeps every token it has run. When the new prompt starts with those
tokens, the reply runs only the tokens that follow, so a long chat
stays fast. The template can render earlier turns differently once a
new turn follows. LFM2's template drops an earlier answer's thinking,
for example. LFM2's convolution state cannot rewind to a shared prefix,
so such a prompt runs from a fresh state.

Greedy decoding runs the tokens it picks in chunks of 8, after a first
chunk of one token that starts the reply at once. The GPU also runs the
tokens after the stop token in the last chunk, so the next greedy
prompt usually runs from a fresh state. Sampled decoding runs one token
at a time and keeps its state.

A failed reply in `bobcat chat`, such as one that overflows the context,
prints the error and leaves the conversation open.

LFM2.5 models can think before they answer. The model writes its
thinking between the single tokens `<think>` and `</think>`. The
conversation recognizes those tokens by id and tags each piece of text
as thinking or answer. `bobcat respond` sends the thinking to standard
error or drops it. `bobcat chat` shows it dimmed.

## Sampling

`crates/bobcat-cli/src/sampler.rs` picks each next token from the logits.
The steps run in llama.cpp's order.

| Step | Flag | Effect |
|---|---|---|
| Repetition penalty | `--repeat-penalty` | Divides the positive logits of the last 64 distinct tokens by the penalty and multiplies their negative logits by it |
| Top-k | `--top-k` | Keeps the k most likely tokens |
| Top-p | `--top-p` | Keeps the fewest tokens whose probabilities sum to p |
| Min-p | `--min-p` | Drops tokens less likely than p times the most likely token |
| Temperature | `--temperature` | Sharpens or flattens the distribution before the draw |
| Draw | `--seed` | Picks one token at random from what remains |

The defaults come from the model file's `general.sampling.*` metadata,
then from Liquid's model cards.

| Setting | Default |
|---|---|
| Temperature | 0.1 |
| Top-k | 50 |
| Top-p | 1.0 |
| Min-p | 0.0 |
| Repetition penalty | 1.1 for models over 2 billion weights, such as LFM2.5-2.6B, and 1.05 for smaller ones |

A temperature of zero with no repetition penalty is greedy decoding.
Greedy decoding lets the GPU pick each token itself and run several
steps ahead of the CPU, which the [Metal backend](metal.md) describes.
Any other setting reads each step's logits back to the CPU.
