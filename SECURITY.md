# Security policy

bobcat reads model files that people download from strangers, so we treat
every parsing and memory-safety bug as a security bug. Thank you for
helping keep bobcat and its users safe.

## Reporting a vulnerability

Please report vulnerabilities privately, and do not open a public issue
or pull request about them.

1. Open the repository's
   [Security tab](https://github.com/jadidbourbaki/bobcat/security) and
   choose **Report a vulnerability**. GitHub keeps the report private
   between you and the maintainers.
2. If you cannot use GitHub, email hayder@alumni.harvard.edu.

A useful report describes the vulnerability, the steps or file that
reproduce it, the impact you expect, and a fix if you have one. A
crafted GGUF file that triggers the bug is the most helpful
reproduction.

## What to expect

| Step | Time |
|---|---|
| Acknowledgement of your report | within 2 business days |
| A first assessment and a plan | within 7 business days |
| A fix | depends on severity and complexity |

We coordinate the disclosure date with you. After a fix ships, we
publish a GitHub security advisory and credit you by name unless you
would rather stay anonymous.

## Supported versions

bobcat has not reached 1.0. Security fixes go into the latest release
only.

## Scope

The following count as vulnerabilities:

- A GGUF file or `tokenizer.json` that makes bobcat read or write memory
  out of bounds, crash, hang, or allocate without bound.
- A model name or download that makes `bobcat pull` write outside the
  Hugging Face cache.
- A problem in `install.sh` or the release binaries that lets someone
  other than the maintainers change what gets installed.

bobcat runs the model the user chooses. Harmful or false text that a model
produces is a property of that model and outside this policy.

## How bobcat limits the damage

- The GGUF parser in `crates/bobcat-gguf` contains no unsafe code, and its
  lints reject unchecked indexing and arithmetic, so a malformed file
  yields an error.
- Unsafe code is confined to the Metal backend, memory-mapping model
  files, and one thread-priority call, as `AGENTS.md` lists. Every
  kernel launch checks that the memory it binds lies inside its
  buffer.
- `install.sh` checks the SHA-256 checksum of the binary it downloads
  before installing it.
