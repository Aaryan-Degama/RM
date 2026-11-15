"""T4: BLINK and TIGER re-run under TIGER's protocol (C1), and the same lists after B with TempEL's
knobs unchanged (c from each model's own validation lists). 2019 new-entity models, test 2019-2022,
with and without titles in the eval pool.

Reads the Kaggle dumps (code/kaggle_c1.sh) and TIGER's released files (Zenodo 10977757, in data/tiger/ext).
Reuses rerank.prep by swapping its TempEL loaders for TIGER's pool. One pass: refuses to overwrite.
  PYTHONPATH=code .venv/bin/python code/tiger_b.py
"""
import json
import os

import numpy as np
import torch
from transformers import BertTokenizer

import rerank
from rerank import Lists, entity_stats, fold_ranks

K, T, OUT = 'data/kaggle/dumps', 'data/tiger/ext', 'data/kaggle/tiger_b.json'
ROWS = ('recency', 'snapshot-year', 'title-matcher', 'B')
knobs = json.load(open('data/enc/knobs_d64.json'))
tok = BertTokenizer('data/tempel/2019/vocab.txt', do_lower_case=True)
_window = rerank.window_slots
# TIGER's mentions are raw strings with pre-tokenized contexts; same 128-token BLINK window
rerank.window_slots = lambda row, vocab: _window({'mention_bert': tok.tokenize(row['mention']),
                                                  'context_left_bert': row['context_left'],
                                                  'context_right_bert': row['context_right']}, vocab)


def lists(run, year):
    prefix = f'data/kaggle/prep/{run}-{year}'
    if not os.path.exists(prefix + '.fam64.npz'):
        os.makedirs('data/kaggle/prep', exist_ok=True)
        d = torch.load(f'{K}/{run}/{year}_test.pt', weights_only=True)
        docs = [json.loads(l) for l in open(f'{T}/01_documents_no_duplicates/{year}_test.json', encoding='utf-8')]
        jsonl = f'{T}/new_entities_1764/{year}/test.jsonl'
        q = np.array([int(x['qid'][1:]) for x in docs])  # pool index = line number
        assert len(set(q.tolist())) == len(q) == len(d['pool_norm'])
        title = {int(x['qid'][1:]): x['title'] for x in docs}
        rerank.label_qids = lambda: q
        rerank.page_titles = lambda need: {k: title[k] for k in need}
        rerank.entity_norms = lambda: d['pool_norm'].numpy()
        rerank.SNAP_YEAR = year - 1
        ids, gold = d['ids'].numpy(), d['gold'].numpy()
        hit = ids == gold[:, None]
        # gold outside the top-64: rank unknown, so MRR below is MRR@64
        np.savez(prefix + '.top256.npz', ids=ids, scores=d['scores'].numpy(), gold=gold,
                 gold_rank=np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 9))
        with open(prefix + '.meta.jsonl', 'w', encoding='utf-8') as f:
            for i, l in enumerate(open(jsonl, encoding='utf-8')):
                x = json.loads(l)
                assert q[gold[i]] == int(x['qid'][1:]), (i, 'dump gold is not the mention gold')
                f.write(json.dumps({'line': i, 'category': x['category'], 'target_qid': x['qid'],
                                    'target_title': x['label_title'], 'mention': x['mention']}) + '\n')
        rerank.prep(prefix, jsonl, 64)
    return Lists(prefix, 64)


def main():
    assert not os.path.exists(OUT), f'{OUT} exists: one pass (delete it only to rerun deliberately)'
    res, frozen = {}, {}
    for cfg in ('baseline', 'paper'):
        v = torch.load(f'{K}/{cfg}-new_entities-2019-val/2019_validation.pt', weights_only=True)['scores']
        c = float((v[:, 0] - v[:, 1]).median())
        for suffix in ('', '-title'):  # no validation dump with titles: the no-title c is used
            for year in (2019, 2020, 2021, 2022):
                L = lists(f'{cfg}-new_entities-2019{suffix}', year)
                o0 = L.order()
                assert (o0 == np.arange(L.D)).all()
                r0, t0 = L.ranks(o0)
                frozen[cfg, suffix, year] = (L, r0)
                cc = L.cont & L.changeable
                out = {'c': c, 'frozen_new': L.metrics(r0, t0, L.new, r0), 'frozen_continual': L.metrics(r0, t0, L.cont, r0),
                       'pooled_acc@1': float((r0 == 1).mean()), 'oracle_new_acc@1': float((L.oracle()[L.new] == 1).mean()),
                       'new_gold_in_family': float((L.gold_fam[L.new] >= 0).mean()), 'n_changeable_continual': int(cc.sum())}
                for name in ROWS:
                    r, t = fold_ranks(L, {**knobs, 'c': c}, name)
                    out[name] = {'new': L.metrics(r, t, L.new, r0), 'delta_new': entity_stats(L, r0, r, L.new),
                                 'changeable_continual': entity_stats(L, r0, r, cc), 'continual': L.metrics(r, t, L.cont, r0)}
                res[f'{cfg}{suffix}|{year}'] = out
                print(f'{cfg}{suffix} {year}: new acc@1 {out["frozen_new"]["acc@1"]:.4f} -> B {out["B"]["new"]["acc@1"]:.4f} '
                      f'{[round(x, 4) for x in out["B"]["delta_new"]]} | oracle {out["oracle_new_acc@1"]:.4f} | '
                      f'cc {[round(x, 4) for x in out["B"]["changeable_continual"]]}', flush=True)
    # C1: TIGER minus BLINK on the same mentions (entity-level bootstrap CI, sign-flip p)
    for suffix in ('', '-title'):
        for year in (2019, 2020, 2021, 2022):
            (L, rb), (_, rt) = frozen['baseline', suffix, year], frozen['paper', suffix, year]
            res[f'tiger-blink{suffix}|{year}'] = {k: entity_stats(L, rb, rt, s) for k, s in
                                                  (('pooled', L.new | L.cont), ('new', L.new), ('continual', L.cont))}
            print(f'TIGER - BLINK{suffix} {year}:', {k: [round(x, 4) for x in v] for k, v in res[f'tiger-blink{suffix}|{year}'].items()})
    json.dump({'knobs': knobs, 'results': res}, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
