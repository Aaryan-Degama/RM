#!/usr/bin/env bash
# Fetch TempEL files one by one (resumable). Never use TempEL's own download
# scripts: they zip the whole 2.2 TB share.  Usage: code/download.sh small|tensors
set -euo pipefail
B=https://cloud.ilabt.imec.be/public.php/dav/files
H='X-Requested-With: XMLHttpRequest'
OUT=data/tempel
small=(
  i7dWxc4Fkw7cKcC/2019/vocab.txt
  DSiKXQwWpQacRJ6/2019/2021-01-01T00:00:00Z_label_id_to_wikidata_qid_mappings.pt
  RinXy8NgqdW58RW/tempel_v1.0_only_bert_tokenized/test/2021-01-01T00:00:00Z_test.jsonl
  RinXy8NgqdW58RW/tempel_v1.0_only_bert_tokenized/validation/2021-01-01T00:00:00Z_validation.jsonl
  Ytt2WDTJH5r3w4z/stats/snapshot_stats/page_info_2021-01-01T00:00:00Z.csv
  i7dWxc4Fkw7cKcC/2019/model_epoch_09.bin
)
tensors=(DSiKXQwWpQacRJ6/2019/2021-01-01T00:00:00Z_tensors.pt)
declare -n list=$1
for p in "${list[@]}"; do
  f="$OUT/${p#*/}"; mkdir -p "$(dirname "$f")"
  echo "[$(date +%T)] $p"
  # the server caps each connection at ~1.5 MB/s, so use 8 parallel ranges
  aria2c -q -c -x 8 -s 8 -k 64M --file-allocation=none --max-tries=10 --retry-wait=10 \
    --header="$H" -d "$(dirname "$f")" -o "$(basename "$f")" "$B/$p"
done
echo "[$(date +%T)] done $1"
