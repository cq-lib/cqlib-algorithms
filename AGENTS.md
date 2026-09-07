# Repository instructions

- Read `docs/development.md` before making changes.
- Keep the structure minimal; add modules only with concrete implementations.
- Use English for code, comments, and docstrings.
- Do not install or synchronize unpublished SDKs; the developer installs them manually.
- Run `uv run --no-sync pytest` for relevant changes and `uv run --no-sync python -m build` for packaging changes. Report checks actually run and any limitations.
