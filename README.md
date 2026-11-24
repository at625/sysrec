# sysrec: a multi-stage recommender on MovieLens-32M

Work in progress. See `PROGRESS.md` for status and `CLAUDE.md` for the build spec.

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
python scripts/env_check.py
python scripts/download_data.py && python scripts/prepare_data.py && python scripts/data_report.py
pytest
```
