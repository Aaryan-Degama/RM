"""Show the retrieval for one test mention: TempEL's frozen top-10 next to the list after our reorder (B).
Reads the stored lists of the single pre-registered pass: it displays, it does not score anything new.
Run from the repo root:  PYTHONPATH=code .venv/bin/python code/demo.py [word or line number]"""
import json, os, pickle, sys
import numpy as np
from families import page_titles
from rerank import Lists
from retrieve import label_qids

P, TOP = 'data/enc/test2021full', 10
L = Lists(P, 64)
meta = [json.loads(l) for l in open(P + '.meta.jsonl', encoding='utf-8')]
knobs = json.load(open('data/enc/knobs_d64.json'))
cfg = knobs['rows']['B']['0']
assert all(v == cfg for v in knobs['rows']['B'].values())  # every fold chose the same knobs
order = L.order('w', cfg['lam'], cfg['eta'], knobs['c'])
L.check(order)
after, _ = L.ranks(order)
changed = (order != np.arange(L.D)).any(1)
fixed = np.nonzero(L.new & (L.gold_rank > 1) & (after == 1))[0]  # new-entity mentions B corrected
qids = label_qids()
cache = 'data/enc/demo_titles.pkl'
if os.path.exists(cache):
    title = pickle.load(open(cache, 'rb'))
else:
    print('first run: reading page titles (about a minute) ...')
    title = page_titles(set(np.unique(qids[L.ids]).tolist()))
    pickle.dump(title, open(cache, 'wb'))


def show(i):
    m, c = meta[i], meta[i]['ctx']
    at = c.find(m['mention'].lower())
    if at >= 0:
        c = c[max(at - 250, 0):at] + '[[' + c[at:at + len(m['mention'])] + ']]' + c[at + len(m['mention']):at + 250]
    in_list = L.gold_pos[i] >= 0
    print(f"\n--- test mention {i} ({'NEW' if L.new[i] else 'continual'} entity) ---\n...{c[:520]}...")
    print(f"mention: {m['mention']!r}   correct page: {m['target_title'].replace('_', ' ')}")
    print(f"rank of the correct page: {L.gold_rank[i] if in_list else 'not in top-64'} -> {after[i] if in_list else 'unchanged'}"
          f"   (pool: {len(qids):,} pages; >> = correct page, F = in a title family; only these can move)\n")

    def cell(p):
        mark = '>>' if p == L.gold_pos[i] else ' F' if L.fam[i, p] >= 0 else '  '
        return f"{mark} {title[int(qids[L.ids[i, p]])].replace('_', ' ')[:40]:40s} {L.scores[i, p]:6.1f}"

    print(f"{'':4s}{'TempEL model (frozen)':51s}| after our reorder")
    for k in range(TOP):
        print(f'{k + 1:2d}  {cell(k)} | {cell(order[i, k])}')


def find(q):
    if not q:
        return int(np.random.choice(fixed))
    if q.isdigit():
        return int(q) if int(q) < L.M else None
    hits = [j for j, m in enumerate(meta) if q.lower() in (m['mention'] + ' ' + m['target_title']).lower().replace('_', ' ')]
    hits.sort(key=lambda j: (not changed[j], not L.new[j]))  # reordered lists first, then new entities
    print(f'{len(hits)} test mentions match {q!r}' + (f'; others: {hits[1:8]}' if len(hits) > 1 else ''))
    return hits[0] if hits else None


if len(sys.argv) > 1:
    i = find(' '.join(sys.argv[1:]))
    if i is not None:
        show(i)
else:
    print(f'{L.M:,} test mentions loaded; {len(fixed)} new-entity mentions corrected by the reorder.')
    while (q := input('\nword, line number, Enter = a corrected new-entity example, q = quit: ').strip()) != 'q':
        i = find(q)
        if i is not None:
            show(i)
