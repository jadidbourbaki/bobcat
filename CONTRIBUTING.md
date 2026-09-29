# Contributing to bobcat

Thank you for considering a contribution to bobcat. Bug reports, benchmark
results from your Mac, documentation fixes, tests, new model families,
and faster kernels are all welcome, and first contributions are welcome
too. Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## Ways to help

- **Report a bug** with the bug report form on the
  [issues page](https://github.com/jadidbourbaki/bobcat/issues). Include
  your Mac model, your macOS version, the command you ran, and the
  model name.
- **Share benchmark numbers** from your machine. Different chips and
  core counts show where the kernels need work.
- **Suggest a feature or a model family** with the feature request
  form.
- **Send a pull request.** For anything larger than a small fix, open
  an issue first, so we can agree on the approach before you spend
  time on it.

## Setting up

bobcat builds on a Mac with Apple silicon and macOS 15 or newer. It needs
no Xcode.

1. Install Rust with [rustup](https://rustup.rs). The repository pins
   its Rust version in `rust-toolchain.toml`, and rustup installs that
   version on first use. If Homebrew's `rust` is also installed, put
   `~/.cargo/bin` first in your `PATH`.
2. Install the tools the checks use:

   ```console
   $ brew install just clang-format
   $ cargo install --locked cargo-sort
   ```

3. Fork the repository, clone your fork, and run the checks:

   ```console
   $ git clone https://github.com/YOUR_USERNAME/bobcat.git
   $ cd bobcat
   $ just check
   ```

`just check` runs the formatters, clippy, the documentation build, and
the tests. `just` lists the other recipes.

### Test models

The reference tests compare bobcat's forward pass, layer by layer, with
Hugging Face transformers. They need model files in `models/` and
reference dumps in `models/ref/`, and they print `skip:` when a file is
missing. To produce the dumps, install [uv](https://docs.astral.sh/uv/)
and run `tools/ref_dump.py`, whose docstring gives the commands.

## Making a change

1. Create a branch from `main`, named `your_username/short_description`.
2. Make the change, with a test when it fixes a bug or adds behavior.
3. Run `just check` until it passes.
4. Commit with a one-line [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)
   message:

   ```text
   feat: add Q5_K matrix-vector kernels
   fix: reject GGUF tensors whose offset exceeds the file size
   perf: fuse RMSNorm into the following projection
   docs: explain model names in the README
   ```

5. Push to your fork and open a pull request. The template lists what
   reviewers look for.

`AGENTS.md` holds the project's conventions: the Rust style, the unsafe
code policy, the prose style for comments and docs, and the benchmark
protocol. Reviewers point to it, so a skim before a first pull request
saves a round of review.

### Performance changes

A change that claims a speedup includes before and after numbers from
`bobcat-bench`, measured on an idle Mac, with the chip named. A change
that slows any benchmark says so. `AGENTS.md` describes the protocol.

## Using AI tools

You may use any tools you like, including AI assistants, as long as a
person stays in the loop:

- Read, understand, and test everything you submit. You are the author
  of your contribution and answer for it in review.
- Be ready to answer questions about your change yourself.
- Say in the pull request when AI tools wrote a significant part of it.

Unreviewed AI output shifts the work of reviewing it onto the
maintainers, so we close pull requests that show no sign of human
review. New contributors do best starting small, with changes they
understand fully.

## Recognition

Every contributor appears in the GitHub contributors list and in the
release notes of the version their change ships in.
