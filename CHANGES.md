# Changelog

## [2025-12-01] - The "Premium" Overhaul

### Added
- **`src/config.py`**: Centralized all configuration (paths, params, constants). No more hardcoded magic strings!
- **Interactive Dashboard**: Replaced Streamlit with a custom Flask + Plotly app (`app.py`, `templates/`, `static/`).
- **`CHANGES.md`**: This file!

### Changed
- **Refactored All Scripts**: Updated `src/*.py` to use `config.py` and PySpark where appropriate.
- **Humanized Comments**: Rewrote code comments to be casual and explanatory ("Why" instead of "What").
- **Updated `run.sh`**: Simplified orchestration to rely on config defaults.
- **README**: Completely rewritten to be friendlier and more useful.

### Removed
- **`django_dashboard/`**: Deleted the unused Django project.
- **`streamlit`**: Removed dependency and code references.
- **Dead Code**: Cleaned up unused imports and legacy comments.
