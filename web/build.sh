#!/bin/bash
# Bundle the volcorr package (code + small tables, no model files) for the static web page.
set -euo pipefail
cd "$(dirname "$0")/../src"
rm -f ../web/volcorr.zip
zip -qr ../web/volcorr.zip volcorr -x '*/__pycache__/*' '*.txt.gz' '*_mlp.npz'
echo "web/volcorr.zip: $(du -h ../web/volcorr.zip | cut -f1)"
