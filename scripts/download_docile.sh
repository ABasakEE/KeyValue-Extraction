#!/bin/bash
# Download one DocILE part with the token from .secrets/docile_token (gitignored), without printing the token or the URL.
# usage: scripts/download_docile.sh annotated-trainval [data/docile]     (names as in the docile README: annotated-trainval, test, synthetic, ...)
set -euo pipefail
ds="$1"; dir="${2:-data/docile}"; mkdir -p "$dir"
tok=$(tr -d '\r\n ' < "$(dirname "$0")/../.secrets/docile_token")
curl -sS -L -C - --fail -o "$dir/$ds.zip" "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/$tok/$ds.zip"
[ "$(wc -c < "$dir/$ds.zip")" -gt 100000000 ] || { echo "unexpected size of $ds.zip"; exit 1; }
python -I -m zipfile -e "$dir/$ds.zip" "$dir" && rm "$dir/$ds.zip"
echo "done: $ds -> $dir"
