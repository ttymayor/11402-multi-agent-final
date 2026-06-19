# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Streamlit application. The main application, agent classes, retrieval helpers, quiz parsing, and UI rendering live in `app.py`. Project metadata and dependencies are defined in `pyproject.toml`, with locked versions in `uv.lock`. `README.md` is currently empty. There is no committed test suite yet; add future tests under `tests/`, mirroring the functions or behaviors being covered, for example `tests/test_quiz_parsing.py`.

## Build, Test, and Development Commands

- `uv sync`: create or update the local `.venv` from `pyproject.toml` and `uv.lock`.
- `uv run streamlit run app.py`: start the local Streamlit app.
- `uv run python -m compileall app.py`: quick syntax check for the current single-file app.
- `uv run pytest`: run tests once a `tests/` directory and `pytest` dependency are added.

The project requires Python `3.14` as specified in `.python-version` and `pyproject.toml`.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, descriptive snake*case functions, and PascalCase classes. Keep Streamlit UI helpers named with `render*`when they produce interface elements, and keep agent orchestration classes grouped near related prompt constants. Prefer type hints for new helpers, following the existing style such as`parse_count(raw: str) -> int | None`. Avoid broad refactors in `app.py` unless splitting the module is part of the task.

## Testing Guidelines

Prioritize tests for deterministic logic before UI behavior: quiz request parsing, JSON normalization, count validation, chunking, tokenization, and retrieval ranking. Name test files `test_*.py` and test functions `test_*`. For GenAI-dependent flows, isolate API calls behind fakes or mocks so tests do not require network access or real credentials.

## Security & Configuration Tips

The Gemini API key is entered at runtime in the Streamlit sidebar. Do not commit API keys, local `.venv` files, generated caches, or uploaded course materials. Keep `.gitignore` updated if new generated directories are introduced.
