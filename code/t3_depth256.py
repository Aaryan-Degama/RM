"""T3: the depth-64 knobs applied unchanged to the top-256 lists (one test pass, pre-registered item 6).
Re-ranking 256 candidates can change acc@64, TempEL's published metric.
  PYTHONPATH=code python code/t3_depth256.py     (needs rerank.py prep ... --depth 256)"""
import json, os
import numpy as np
from rerank import Lists, fold_ranks, entity_stats

P, OUT = 'data/enc/test2021full', 'data/enc/test2021full.t3_d256.json'
assert not os.path.exists(OUT), f'{OUT} exists: one test pass per row'
knobs = json.load(open('data/enc/knobs_d64.json'))
assert knobs['tuned_on'] != P
L = Lists(P, 256)
o0 = L.order()
assert (o0 == np.arange(L.D)).all()  # G4
r0 = L.ranks(o0)[0]
ro = L.oracle()
res = {}
for name in ('B', 'snapshot-year', 'recency', 'title-matcher'):
    r = fold_ranks(L, knobs, name)[0]  # asserts G1, G2
    for pop, sel in (('new', L.new), ('continual', L.cont)):
        row = {'n': int(sel.sum())}
        for k in (1, 8, 64, 256):
            at = lambda x: np.where(x <= k, 1, 2)  # entity_stats compares rank == 1
            row[f'acc@{k}'] = {'frozen': float((r0[sel] <= k).mean()), 'row': float((r[sel] <= k).mean()),
                               'oracle': float((ro[sel] <= k).mean()), 'delta_ci_p': entity_stats(L, at(r0), at(r), sel)}
        res[f'{name}|{pop}'] = row
        print(name, pop, ' '.join(f"@{k} {row[f'acc@{k}']['frozen']:.4f}->{row[f'acc@{k}']['row']:.4f} "
                                  f"(oracle {row[f'acc@{k}']['oracle']:.4f})" for k in (1, 8, 64, 256)),
              'd@64 %+.2f [%+.2f, %+.2f] p=%.4f' % tuple(100 * v if i < 3 else v for i, v in enumerate(row['acc@64']['delta_ci_p'])))
json.dump(res, open(OUT, 'w'), indent=1)
