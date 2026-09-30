# AGENTS.md

Guidance for AI agents working on bobcat's Python tools. The CLAUDE.md
symlink resolves to this file. The repository's root `AGENTS.md`
applies here too, including its prose rules, git practices, and the
rule to research before building.

Python appears only in `tools/`, for reference dumps, benchmark
scripts, figures, and the mascot. The engine never depends on Python.
`tools/` is one uv project with its own `pyproject.toml` and
`uv.lock`.

## Tooling

The toolchain is Astral's.

- **uv** manages environments and dependencies. Add a dependency with
  `uv add`, resolve with `uv lock`, and run a script with `uv run`.
  pip, poetry, and `requirements.txt` have no place in the repo.
- **ruff** lints and formats. `pyproject.toml` sets the line length to
  100, the same width as the Rust code, and selects the `E`, `F`, `I`,
  `B`, `UP`, and `SIM` rules.
- **ty** type-checks. mypy and pyright stay out of the gate.
- `just check` at the repository root runs `ruff format --check`,
  `ruff check`, and `ty check` over `tools/`. `just fmt` runs
  `ruff format`.

## Dependencies

- Runtime dependencies go in `[project].dependencies`. ruff and ty go
  in `[dependency-groups].dev`, which `uv run` installs by default.
- Pin every direct dependency to an exact version with `==` and
  commit `uv.lock`. The lockfile pins the whole tree with hashes, so
  every install is reproducible and verified. uv writes the lockfile,
  and nobody edits it by hand.
- `.gitignore` covers `.venv/`, `__pycache__/`, `.ruff_cache/`, and
  `.ty_cache/`.

## Types

- Put `from __future__ import annotations` at the top of every
  module.
- Type every function signature, both parameters and return.
- Use built-in generics such as `list[int]` and `dict[str, float]`,
  and write `X | None` for an optional value.
- Give structured data a model. Data read from outside the process,
  such as a CSV file, a JSON file, or command output, gets a Pydantic
  model that validates it at the edge. Data that stays inside the
  script gets a frozen dataclass. A signature such as
  `list[dict[str, Any]]` means the rows need a model.

## Imports and names

- Every import goes at the top of the module. Import inside a
  function only to keep an optional dependency optional, and say so
  in a comment.
- Use `snake_case` for functions and variables, `PascalCase` for
  classes, and `UPPER_SNAKE` for module constants. A leading
  underscore marks a name private to its module.

## Errors

- Raise an exception on failure. A function never returns `None`,
  `""`, or `-1` to mean that it failed.
- Catch narrowly, and never write a bare `except:`. A script that hits
  a bad state stops with a traceback.
- Re-raise with `raise ... from err`, so the cause survives.

## Comments

The root `AGENTS.md` prose rules apply to Python comments and
docstrings.

- Open each script with a docstring that serves as its usage message:
  what the script does, the command that runs it, and what it writes.
- Give a function a docstring when its contract is not obvious from
  its name and signature.
- Comment only the non-obvious why, such as a measured constant or a
  workaround.
