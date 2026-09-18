set -euo pipefail
SRC=https://github.com/ahmed-shameem/Feature_selection.git
COMMIT=257d7dca4ae643733a69dcf06cb7c70b7cce37e2
TMP=$(mktemp -d)
git clone -q "$SRC" "$TMP/src"
git -C "$TMP/src" checkout -q "$COMMIT"
mkdir -p data/raw
cp "$TMP"/src/UCI_datasets/*.csv data/raw/
python scripts/make_manifest.py > /dev/null
git diff --exit-code data/manifest.csv && echo "Data fetched; checksums match data/manifest.csv"
