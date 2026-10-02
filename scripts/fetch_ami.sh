#!/usr/bin/env bash
# Fetch the AMI Meeting Corpus manual annotations into data/ami/.
# ~22 MB. Not redistributed with this repository — see README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"
URL="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ZIP="$DEST/ami_public_manual_1.6.2.zip"

mkdir -p "$DEST"

if [ -d "$DEST/ami/words" ]; then
  echo "already present: $DEST/ami"
  exit 0
fi

echo "downloading AMI manual annotations (~22 MB)…"
curl -L --fail -C - -o "$ZIP" "$URL"

echo "unpacking…"
mkdir -p "$DEST/ami"
unzip -q -o "$ZIP" -d "$DEST/ami"
rm -f "$ZIP"

echo
echo "done. $(find "$DEST/ami/words" -name '*.xml' | wc -l) word files in $DEST/ami"
echo "run:  python src/run.py"
