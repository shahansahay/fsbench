set -euo pipefail
python scripts/make_manifest.py
pytest -q
python scripts/build_tables.py
python scripts/verify_tables.py
python scripts/run_experiments.py --workers 8
python scripts/analyze.py
python scripts/extra_stats.py
python scripts/figures.py
python scripts/run_supplement.py --workers 8
python scripts/analyze_supplement.py
python scripts/check_reproducibility.py
