#!/bin/sh
# The anonymized review mirror cannot serve files larger than 8 MB, so the
# larger JSON files are stored gzip-compressed. Run this once from the
# repository root before running any script; it keeps the .gz files.
set -e
find . -name "*.json.gz" | while read -r f; do
  out="${f%.gz}"
  [ -f "$out" ] || { gunzip -c "$f" > "$out"; echo "unpacked $out"; }
done
