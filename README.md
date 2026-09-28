# gip

<p>
  <a href="https://github.com/jadidbourbaki/gip/actions/workflows/ci.yml"><img src="https://github.com/jadidbourbaki/gip/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/jadidbourbaki/gip/releases/latest"><img src="https://img.shields.io/github/v/release/jadidbourbaki/gip?include_prereleases" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
</p>

**gip runs language models on your Mac's GPU.**

It reads the GGUF files on Hugging Face and runs them with its own
Metal kernels, built to be the fastest way to run a model locally on
Apple silicon.

```console
$ gip chat -m lfm2.5:1.2b
> What is the capital of Japan? One sentence.
The capital of Japan is Tokyo.
```

gip is early. It runs Liquid AI's LFM2 and LFM2.5 models today, in the
Q8_0, Q4_0, and Q4_K_M quantizations. More model families come next.

## Quick start

Install gip on a Mac with Apple silicon and macOS 15 or newer:

```console
$ curl -fsSL https://raw.githubusercontent.com/jadidbourbaki/gip/main/install.sh | sh
```

Download a model:

```console
$ gip pull lfm2.5:1.2b
```

Chat with it:

```console
$ gip chat -m lfm2.5:1.2b
```

Or answer one prompt, for scripts and pipes:

```console
$ gip respond -m lfm2.5:1.2b "Name three primes."
```
