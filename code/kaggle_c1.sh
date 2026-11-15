#!/usr/bin/env bash
# C1 on a Kaggle GPU notebook (Settings: Accelerator = GPU T4, Internet = on).
# Paste this whole file into one cell that starts with %%bash, then "Save Version" ->
# "Save & Run All (Commit)" so it runs in the background (~3-5 h; Kaggle limit 12 h).
#
# Full TIGER grid (student's decision): {baseline.yaml = BLINK, paper.yaml = TIGER} x
# {new_entities, continual_entities} x train years 2019-2022 = 16 trainings, each evaluated
# forward on test years train..2022. Also: 2019 validation dumps (for c and the B adapter) and
# the "titles in the eval pool" control for the 2019 new_entities models.
# Every evaluation dumps top-64 ids/scores, gold and pool squared norms, so pooled, new-only and
# continual-only recall are computed afterwards from the same pass.
# Run 1 (2026-11-15) died on the 16th checkpoint: /kaggle/working is capped at 20 GB. So code, data
# and checkpoints now live in /kaggle/temp, non-2019 checkpoints are deleted after use, and the
# dumps no longer carry mention/pool vectors (~25 MB per file instead of ~115 MB).
# Outputs in /kaggle/working: c1_results.json, c1_table.txt, dumps/ (download these).
set -euo pipefail
COMMIT=649bb198f9569dd3a9d26a3498ac183eebf43290   # TIGER HEAD on 2026-11-06
WORK=/kaggle/working
cd $WORK 2>/dev/null || { WORK=$(pwd); }
SCRATCH=/kaggle/temp; mkdir -p $SCRATCH 2>/dev/null || SCRATCH=$WORK
cd $SCRATCH; df -h $WORK $SCRATCH
[ -d tiger ] || git clone -q https://github.com/pengyu-zhang/TIGER-Temporally-Improved-Graph-Entity-Linker.git tiger
cd tiger && git checkout -q $COMMIT

# Patch evaluate_year: dump top-64 ids/scores, gold, pool squared norms to $DUMP_DIR/<year>_<split>.pt
python - <<'EOF'
p = 'src/tiger/evaluate.py'
s = open(p).read()
if 'DUMP_DIR' not in s:
    for a, b in [
        ("    stats = Stats(top_k)\n",
         "    stats = Stats(top_k)\n    dump = {'ids': [], 'scores': [], 'gold': []}\n"),
        ("        _, top_ids = scores.topk(min(top_k, pool_size), dim=1)\n",
         "        top_s, top_ids = scores.topk(min(top_k, pool_size), dim=1)\n"
         "        dump['ids'].append(top_ids.int().cpu()); dump['scores'].append(top_s.cpu())\n"
         "        dump['gold'].append(label_ids.cpu())\n"),
        ("    return stats, pool_size, len(dataset)\n",
         "    if os.environ.get('DUMP_DIR'):\n"
         "        os.makedirs(os.environ['DUMP_DIR'], exist_ok=True)\n"
         "        tag = '_'.join(test_path.split(os.sep)[-2:]).replace('.jsonl', '')\n"
         "        out = {k: torch.cat(v) for k, v in dump.items()}\n"
         "        out['pool_norm'] = (pool_enc.float() ** 2).sum(1).cpu()\n"
         "        torch.save(out, os.path.join(os.environ['DUMP_DIR'], f'{tag}.pt'))\n"
         "    return stats, pool_size, len(dataset)\n"),
    ]:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    open(p, 'w').write(s)
print('patched', p)
EOF

pip install -q -r requirements.txt
python -c "import torch; x = torch.randn(512, 512, device='cuda'); print('torch', torch.__version__, torch.cuda.get_device_name(0), float((x @ x).sum()))"
YEARS=2019,2020,2021,2022 bash scripts/prepare_data.sh
bash scripts/smoke_test.sh
rm -rf outputs/smoke

ev() {  # ev <cfg> <model_dir> <dump_dir> <split> <years> [extra --set ...]
  local cfg=$1 md=$2 dd=$3 split=$4 years=$5; shift 5
  DUMP_DIR=$dd PYTHONPATH=src python -m tiger.evaluate --config configs/$cfg.yaml --model-dir $md \
    --test-years $years --split $split --set eval.cache_pool_encodings=false "$@"
}
for cfg in baseline paper; do
  for var in new_entities continual_entities; do
    for y in 2019 2020 2021 2022; do
      md=outputs/$cfg/$var/$y
      [ -f $md/pytorch_model.bin ] || bash scripts/train.sh configs/$cfg.yaml $var $y $md
      years=$(python -c "print(','.join(str(t) for t in range($y, 2023)))")
      [ -d $WORK/dumps/$cfg-$var-$y ] || ev $cfg $md $WORK/dumps/$cfg-$var-$y test $years
      [ $y = 2019 ] || rm -f $md/pytorch_model.bin   # only the 2019 models are needed again
    done
    ev $cfg outputs/$cfg/$var/2019 $WORK/dumps/$cfg-$var-2019-val validation 2019
  done
  ev $cfg outputs/$cfg/new_entities/2019 $WORK/dumps/$cfg-new_entities-2019-title test 2019,2020,2021,2022 \
     --set data.use_title_in_pool=true
done

python - <<'EOF'
import glob, json, os, torch
W = '/kaggle/working' if os.path.isdir('/kaggle/working') else '..'
ks = (1, 2, 4, 8, 16, 32, 64)
res = {}
for d in sorted(glob.glob(f'{W}/dumps/*')):
    run = os.path.basename(d)
    for f in sorted(glob.glob(f'{d}/*_test.pt')) + sorted(glob.glob(f'{d}/*_validation.pt')):
        year, split = os.path.basename(f)[:-3].split('_')
        x = torch.load(f)
        hit = x['ids'] == x['gold'][:, None].int()
        rank = torch.where(hit.any(1), hit.int().argmax(1), torch.full_like(x['gold'], -1)).tolist()
        cats = [json.loads(l)['category'] for l in open(f'data/processed/graph_tempel/mentions/all/{year}/{split}.jsonl') if l.strip()]
        assert len(rank) == len(cats), (f, len(rank), len(cats))
        for name, keep in (('pooled', None), ('new', 'new_entities'), ('continual', 'shared')):
            r = [v for v, c in zip(rank, cats) if keep is None or c == keep]
            res[f'{run}|{split}|{year}|{name}'] = {'n': len(r), **{f'R@{k}': sum(0 <= v < k for v in r) / len(r) for k in ks}}
json.dump(res, open(f'{W}/c1_results.json', 'w'), indent=1)

# forward-only gap means (TIGER/CYCLE Table 2 convention): gap = test year - train year
lines = []
for cfg, label in (('baseline', 'BLINK'), ('paper', 'TIGER')):
    for var, short in (('continual_entities', 'C'), ('new_entities', 'N')):
        for name in ('pooled', 'new', 'continual'):
            row = []
            for gap in range(4):
                v = []
                for k in res:
                    run, split, year, part = k.split('|')
                    c, vv, *rest = run.split('-')  # main runs are cfg-var-trainyear
                    if (c, vv) == (cfg, var) and len(rest) == 1 and split == 'test' and part == name \
                            and int(year) - int(rest[0]) == gap:
                        v.append(res[k]['R@1'])
                row.append(f'{sum(v) / len(v):.3f}' if v else '  -  ')
            lines.append(f'{label:5s} {short} {name:9s} R@1 gaps 0-3: ' + '  '.join(row))
published = 'published (TIGER/CYCLE Table 2, R@1, trained on N, pooled test): BLINK .132 .132 .132 .142 | TIGER .186 .195 .188 .217 | CYCLE .184 .191 .192 .222'
open(f'{W}/c1_table.txt', 'w').write('\n'.join(lines + [published]) + '\n')
print('\n'.join(lines + [published]))
EOF
df -h $WORK $SCRATCH; du -sh $WORK/dumps
