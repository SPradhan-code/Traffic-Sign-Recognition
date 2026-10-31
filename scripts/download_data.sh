#!/usr/bin/env bash
# Downloads GTSRB from Kaggle into data/raw. Requires the Kaggle CLI + API token:
#   pip install kaggle ; put kaggle.json in ~/.kaggle/ (Kaggle > Settings > Create New Token)
set -euo pipefail
cd "$(dirname "$0")/.."
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p data/raw --unzip
echo "Done. Expect data/raw/Train/0..42, data/raw/Test/, data/raw/Test.csv"
ls data/raw
