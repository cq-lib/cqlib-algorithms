# Repository instructions

- Read `docs/development.md` before making changes.
- Follow `docs/documentation-style.md` when adding or changing public documentation and docstrings.
- Keep the structure minimal; add modules only with concrete implementations.
- Use English for code, comments, and docstrings.
- Use the supported dependency versions declared in `pyproject.toml`.
- Run `uv run --no-sync pytest` for relevant changes and `uv run --no-sync python -m build` for packaging changes. Report checks actually run and any limitations.
