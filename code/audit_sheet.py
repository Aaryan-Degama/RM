"""Blind hand-audit sheet: 100 seeded frozen-model errors on new test entities (gold in top-64).
Writes data/audit/audit_sheet.csv (to label) and data/audit/audit_key.csv (automatic bucket; do not open before labelling)."""
import csv, json
import numpy as np
from families import page_titles
from retrieve import label_qids

P = 'data/enc/test2021full'
z = np.load(P + '.fam64.npz')
meta = [json.loads(l) for l in open(P + '.meta.jsonl', encoding='utf-8')]
M = len(meta)
err = np.nonzero(z['new'] & (z['gold_pos'] > 0))[0]
pick = np.sort(np.random.default_rng(0).choice(err, 100, replace=False))
qids = label_qids()
title = page_titles(set(qids[z['ids'][pick, 0]].tolist()))
with open('data/audit/audit_sheet.csv', 'w', newline='', encoding='utf-8') as a, \
        open('data/audit/audit_key.csv', 'w', newline='', encoding='utf-8') as k:
    a, k = csv.writer(a), csv.writer(k)
    a.writerow(['row', 'mention', 'context', 'correct_page', 'model_picked', 'label (E/U/O/X)', 'note'])
    k.writerow(['row', 'line', 'auto_edition_confusion', 'gold_rank'])
    for n, i in enumerate(pick):
        m = meta[i]
        c, at = m['ctx'], m['ctx'].find(m['mention'].lower())
        c = c[max(at - 300, 0):at + 300] if at >= 0 else c[:600]
        a.writerow([n, m['mention'], c, m['target_title'], title[int(qids[z['ids'][i, 0]])], '', ''])
        k.writerow([n, m['line'], int(z['gold_fam'][i] >= 0 and z['fam'][i, 0] == z['gold_fam'][i]), int(z['gold_pos'][i]) + 1])
print('wrote data/audit/audit_sheet.csv', len(pick), 'rows of', len(err), 'eligible errors')
