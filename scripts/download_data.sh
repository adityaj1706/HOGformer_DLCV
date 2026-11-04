#!/usr/bin/env bash
# Download and arrange the Setting I (All-Weather) data for HOGformer.
# Usage: bash scripts/download_data.sh [DATA_ROOT]     (default DATA_ROOT=./data)
# Result:
#   $DATA_ROOT/weather/train/{gt,input}                      18069 pairs (All-Weather composite)
#   $DATA_ROOT/weather/test/Test1/{gt,input}                 750 pairs   (Outdoor-Rain)
#   $DATA_ROOT/weather/test/RainDrop/test_a/{gt,input}       58 pairs
#   $DATA_ROOT/weather/test/Snow100K-Test/Snow100K-{S,L}/{gt,synthetic}   16611 / 16801 pairs
# Needs ~35 GB free while running (~15 GB at the end). All sources are Google Drive.
set -euo pipefail

ROOT="$(mkdir -p "${1:-./data}" && cd "${1:-./data}" && pwd)"
W="$ROOT/weather"
TMP="$ROOT/_dl"
mkdir -p "$W/train" "$W/test" "$TMP"
pip install -q gdown

# 1) Training set: All-Weather composite (TransWeather / Histoformer)
#    If Google Drive says "Too many users have viewed or downloaded this file",
#    copy the file into your own Drive and pass it in:
#      ALLWEATHER_ZIP=/content/drive/MyDrive/allweather.zip bash scripts/download_data.sh /content/data
#    or (no Drive mount, e.g. Kaggle) pass the Drive file id of your own shared copy:
#      ALLWEATHER_ID=<file id> bash scripts/download_data.sh /tmp/data
if [ ! -d "$W/train/gt" ]; then
  if [ -n "${ALLWEATHER_ZIP:-}" ]; then
    unzip -q -o "$ALLWEATHER_ZIP" -d "$W/train"
  else
    gdown "${ALLWEATHER_ID:-1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl}" -O "$TMP/allweather.zip"
    unzip -q -o "$TMP/allweather.zip" -d "$W/train"
    rm -f "$TMP/allweather.zip"
  fi
fi

# 2) Outdoor-Rain Test1 (link from the Histoformer README)
if [ ! -d "$W/test/Test1" ]; then
  gdown 1xgLG6656ahL7TfpeO_TplkMOQrfdxKbo -O "$TMP/test1.zip"
  unzip -q -o "$TMP/test1.zip" -d "$W/test"
  rm -f "$TMP/test1.zip"
fi

# 3) RainDrop test_a (DeRaindrop Drive folder; we only need the test_a zip)
if [ ! -d "$W/test/RainDrop/test_a" ]; then
  gdown --folder 1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K -O "$TMP/raindrop"
  mkdir -p "$W/test/RainDrop"
  unzip -q -o "$TMP/raindrop/test_a.zip" -d "$W/test/RainDrop"
  rm -rf "$W/test/RainDrop/__MACOSX"
  # The loader pairs files by identical names: data/N_rain.png <-> gt/N_clean.png
  mv "$W/test/RainDrop/test_a/data" "$W/test/RainDrop/test_a/input"
  for f in "$W"/test/RainDrop/test_a/gt/*_clean.png; do
    mv "$f" "${f%_clean.png}_rain.png"
  done
  rm -rf "$TMP/raindrop"
fi

# 4) Snow100K-S / Snow100K-L test sets (Histoformer's mirror; the original
#    Snow100K archive often hits Google Drive's download quota)
# Set SKIP_SNOW=1 to skip this step (not needed for training).
if [ -z "${SKIP_SNOW:-}" ] && [ ! -d "$W/test/Snow100K-Test/Snow100K-L/gt" ]; then
  gdown --folder 1kLPWAev5f6OutKFALhxthuO1XCF5U1Yu -O "$TMP/snow"
  for s in S L; do
    d="$W/test/Snow100K-Test/Snow100K-$s"
    mkdir -p "$d"
    unzip -q -o "$TMP/snow/Snow100K-$s/gt.zip" -d "$d"
    unzip -q -o "$TMP/snow/Snow100K-$s/synthetic.zip" -d "$d"
  done
  rm -rf "$TMP/snow"
fi
rmdir "$TMP" 2>/dev/null || true

# Sanity check: counts and matching names
check() { # dir_gt dir_in expected
  local n1 n2
  n1=$(ls "$1" | wc -l); n2=$(ls "$2" | wc -l)
  if [ "$n1" != "$3" ] || [ "$n2" != "$3" ] || ! diff -q <(ls "$1") <(ls "$2") >/dev/null; then
    echo "MISMATCH: $1 ($n1) vs $2 ($n2), expected $3" >&2; exit 1
  fi
  echo "ok  $3  $1"
}
check "$W/train/gt" "$W/train/input" 18069
check "$W/test/Test1/gt" "$W/test/Test1/input" 750
check "$W/test/RainDrop/test_a/gt" "$W/test/RainDrop/test_a/input" 58
if [ -z "${SKIP_SNOW:-}" ]; then
  check "$W/test/Snow100K-Test/Snow100K-S/gt" "$W/test/Snow100K-Test/Snow100K-S/synthetic" 16611
  check "$W/test/Snow100K-Test/Snow100K-L/gt" "$W/test/Snow100K-Test/Snow100K-L/synthetic" 16801
fi
echo "Data ready under $W"
