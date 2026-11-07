"""Edition-aware within-family re-ranking (C3) and the edition-confusion diagnosis (C2).

Design v3 (docs/ideas/family-reranker.md). The frozen top-D list of each mention is split
into title families (families.families: editions + umbrella page, a partition). Each family
refills exactly the positions it already holds, sorted by (-z, original position) with
    z(e) = s(e) - (lam/2)*||g_e||^2 + eta*c*w(e),
w(e) = typed slot evidence in the encoder's 128-token window (q*rho: q = 1 exact match, 1/2 a
bare year that is a season endpoint; rho = 1 in the mention span, 1/2 elsewhere; non-year slots
count only in the mention span), c = validation median of s1 - s2 (label-free).
Candidates outside families never move (G1); lam = eta = 0 is the frozen list (G4).

  python code/rerank.py prep data/enc/val2021 --jsonl <validation.jsonl>   # -> .fam64.npz
  python code/rerank.py diagnose data/enc/val2021                           # C2 buckets + oracle
  python code/rerank.py tune data/enc/val2021                               # -> data/enc/knobs_d64.json
  python code/rerank.py test data/enc/test2021 --knobs data/enc/knobs_d64.json
"""
import argparse
import json
import os
import re
import subprocess
import zlib

import numpy as np

from encode_mentions import get_context_representation
from families import NUMWORDS, YEAR, families, filter_type, norm, page_titles, slot_of, tau, umbrella_key
from retrieve import entity_norms, label_qids

SNAP_YEAR = 2020  # t - 1 for the 2021 snapshot
ETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 1e6)  # x c; 1e6 = evidence decides
LAMS = (0.0, 0.25, 0.5, 1.0)
TYPES = ('filter-recurrent', 'filter-covid', 'filter-other')
ROMAN_VAL = dict(m=1000, d=500, c=100, l=50, x=10, v=5, i=1)


# ---------------------------------------------------------------- slot evidence
def window_slots(row, vocab):
    """Slots found in the encoder's 128-token window: list of (kind, value, rho, word)."""
    toks = get_context_representation(row, vocab, 128)['tokens']
    i0 = toks.index('[unused0]') if '[unused0]' in toks else -1
    i1 = toks.index('[unused1]') if '[unused1]' in toks else -1
    words = []  # (word, in_mention)
    for i, t in enumerate(toks):
        if t in ('[CLS]', '[SEP]', '[unused0]', '[unused1]'):
            continue
        if t.startswith('##') and words:
            words[-1] = (words[-1][0] + t[2:], words[-1][1])
        else:
            words.append((t, i0 < i < i1))
    # a season is three word pieces: 2019 – 20
    for a, b, c in zip(words, words[1:], words[2:]):
        if b[0] in ('–', '-') and YEAR.fullmatch(a[0]) and re.fullmatch(r'[0-9]{2}|[0-9]{4}', c[0]):
            words.append((a[0] + '–' + c[0], a[1] and c[1]))
    out = []
    for w, inm in words:
        sl = slot_of(w)
        if sl or (inm and (w in NUMWORDS or re.fullmatch(r'[mdclxvi]{2,}', w))):
            kind, val = sl if sl else ('word', w)
            out.append((kind, val, 1.0 if inm else 0.5, w))
    return out


def evidence(slots, ctx):
    """w(e) for a candidate with these title slots, given the window's slots."""
    best = 0.0
    for kind, v in slots:
        for ck, cv, rho, w in ctx:
            q = 0.0
            if kind == 'year':
                q = 1.0 if (ck == 'year' and cv == v) else 0.5 if (ck == 'season' and v in cv) else 0.0
            elif kind == 'season':
                q = 1.0 if (ck == 'season' and cv == v) else 0.5 if (ck == 'year' and cv in v) else 0.0
            elif rho == 1.0:  # non-year slots count only inside the mention span
                if kind in ('ordinal', 'num'):
                    q = 1.0 if (ck in ('ordinal', 'num', 'year') and cv == v) else 0.0
                elif kind in ('roman', 'numword'):
                    q = 1.0 if w == v else 0.0
            best = max(best, q * rho)
    return best


def recency_key(kind, v):
    if kind == 'season':
        return v[0]
    if kind == 'roman':
        vals = [ROMAN_VAL[ch] for ch in v]
        return sum(-x if i + 1 < len(vals) and x < vals[i + 1] else x for i, x in enumerate(vals))
    if kind == 'numword':
        return NUMWORDS.index(v) + 1
    return v


def sib_key(title):
    """Coarse family key: the umbrella name. Two titles are siblings if keys match and one is slotted."""
    tmpl, slots = tau(title)
    return (umbrella_key(tmpl) if slots else norm(title.replace('_', ' ')).lower()), bool(slots)


# ---------------------------------------------------------------- prep (gather pass)
def prep(prefix, jsonl, depth, vocab_path='data/tempel/2019/vocab.txt', seed=0):
    meta = [json.loads(l) for l in open(prefix + '.meta.jsonl', encoding='utf-8')]
    top = np.load(prefix + '.top256.npz')
    ids, scores = top['ids'][:, :depth], top['scores'][:, :depth]
    gold, gold_rank = top['gold'], top['gold_rank']
    M, D = ids.shape
    qids = label_qids()
    need = set(qids[ids].ravel().tolist()) | set(qids[gold[gold >= 0]].tolist())
    title = page_titles(need)
    assert len(title) == len(need), f'{len(need) - len(title)} QIDs without an as-of-t title'
    T = np.vectorize(lambda lab: title[int(qids[lab])], otypes=[object])(ids)
    gtitle = [title[int(qids[g])] if g >= 0 else '' for g in gold]
    nrm = np.asarray(entity_norms())[ids]

    vocab = {t.rstrip('\n'): i for i, t in enumerate(open(vocab_path, encoding='utf-8'))}
    want = {m['line'] for m in meta}
    rows = {}
    with open(jsonl, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i in want:
                rows[i] = json.loads(l)
    ctx = [window_slots(rows[m['line']], vocab) for m in meta]
    del rows

    fam = np.full((M, D), -1, dtype=np.int16)
    w, w_any, rec, snap = (np.zeros((M, D), dtype=np.float32) for _ in range(4))
    for i in range(M):
        for f_idx, members in enumerate(families(list(T[i]))):
            fam[i, members] = f_idx
            slotted = [p for p in members if tau(T[i, p])[1]]
            temporal = [p for p in slotted if any(k in ('year', 'season') for k, _ in tau(T[i, p])[1])]
            pool = temporal or slotted
            key = {p: max(recency_key(k, v) for k, v in tau(T[i, p])[1]
                          if not temporal or k in ('year', 'season')) for p in pool}
            for p in pool:
                rec[i, p] = key[p] == max(key.values())
        for p in range(D):
            slots = tau(T[i, p])[1]
            if slots:
                w_any[i, p] = evidence(slots, ctx[i])
                snap[i, p] = any((k == 'year' and v == SNAP_YEAR) or (k == 'season' and SNAP_YEAR in v)
                                 for k, v in slots)
        w[i] = np.where(fam[i] >= 0, w_any[i], 0)
        snap[i] = np.where(fam[i] >= 0, snap[i], 0)
        if i % 2000 == 0:
            print(f'{i}/{M}', flush=True)

    # placebo: windows shuffled among mentions whose rank-1 candidates share a template
    rng = np.random.default_rng(seed)
    groups = {}
    for i in range(M):
        groups.setdefault(tau(T[i, 0])[0] if fam[i, 0] >= 0 else '', []).append(i)
    # ... always from a mention of a *different* entity, so no real evidence leaks in
    ent = np.array([m['target_qid'] for m in meta])
    perm = np.arange(M)
    for g in groups.values():
        s = rng.permutation(g)
        perm[s] = np.roll(s, 1)
    for i in np.nonzero(ent[perm] == ent)[0]:
        while ent[perm[i]] == ent[i]:
            perm[i] = rng.integers(M)
    assert (ent[perm] != ent).all()
    plac = np.zeros((M, D), dtype=np.float32)
    for i in range(M):
        for p in np.nonzero(fam[i] >= 0)[0]:
            plac[i, p] = evidence(tau(T[i, p])[1], ctx[perm[i]])

    gold_pos = np.where((ids == gold[:, None]).any(1), (ids == gold[:, None]).argmax(1), -1)
    in_list = gold_pos >= 0
    gp = np.maximum(gold_pos, 0)
    gold_fam = np.where(in_list, fam[np.arange(M), gp], -1)
    miss_sib = np.zeros(M, dtype=bool)
    for i in np.nonzero(~in_list & (gold >= 0))[0]:
        gk, gs = sib_key(gtitle[i])
        miss_sib[i] = any(sib_key(t)[0] == gk and (gs or sib_key(t)[1]) for t in T[i])
    gslots = [tau(t)[1] for t in gtitle]
    dated = np.array([any(k in ('year', 'season') for k, _ in s) and
                      not any((k == 'year' and v == SNAP_YEAR) or (k == 'season' and SNAP_YEAR in v) for k, v in s)
                      for s in gslots])
    np.savez(f'{prefix}.fam{depth}.npz', ids=ids, scores=scores, gold=gold, gold_rank=gold_rank,
             fam=fam, n=nrm, w=w, w_any=w_any, rec=rec, snap=snap, plac=plac,
             gold_pos=gold_pos, gold_fam=gold_fam, miss_sib=miss_sib,
             gold_latest=in_list & (gold_fam >= 0) & (rec[np.arange(M), gp] > 0),
             gold_umbrella=in_list & (gold_fam >= 0) & np.array([not s for s in gslots]),
             dated_non_snap=dated,
             fold=np.array([zlib.crc32(sib_key(t)[0].encode()) % 5 for t in gtitle]),
             type=np.array([TYPES.index(filter_type(SNAP_YEAR + 1, m['target_title'])) for m in meta]),
             new=np.array([m['category'] == 'new_entities' for m in meta]) & (gold >= 0),
             cont=np.array([m['category'] == 'shared' for m in meta]) & (gold >= 0),
             entity=np.array([int(m['target_qid'][1:]) for m in meta]))
    print('saved', f'{prefix}.fam{depth}.npz', f'{(fam >= 0).sum()} family members in {M} lists')


# ---------------------------------------------------------------- ordering + metrics
class Lists:
    def __init__(self, prefix, depth):
        self.__dict__.update(np.load(f'{prefix}.fam{depth}.npz'))
        self.M, self.D = self.ids.shape
        self.changeable = self.fam[:, 0] >= 0  # rank-1 lies in an in-list family

    def order(self, ev='w', lam=0.0, eta=0.0, c=1.0, whole_list=False):
        """Order of original positions after re-ranking (M x D)."""
        z = self.scores.astype(np.float64) - lam / 2 * self.n + eta * c * getattr(self, ev)
        if whole_list:  # title-matcher control: no families, sort the whole list
            pos = np.broadcast_to(np.arange(self.D), z.shape)
            return np.lexsort((pos, -z), axis=1)
        m, p = np.nonzero(self.fam >= 0)
        gid = m.astype(np.int64) * self.D + self.fam[m, p]
        by_z = np.lexsort((p, -z[m, p], gid))
        by_pos = np.lexsort((p, gid))
        order = np.tile(np.arange(self.D), (self.M, 1))
        order[m[by_pos], p[by_pos]] = p[by_z]
        return order

    def ranks(self, order):
        """Gold's new rank (frozen rank if outside the list) and the new top-1 position."""
        hit = order == self.gold_pos[:, None]
        r = np.where(self.gold_pos >= 0, hit.argmax(1) + 1, self.gold_rank)
        return r, order[:, 0]

    def check(self, order, whole_list=False):
        """G1, G2 (and G3 via ranks) on every configuration."""
        assert (np.sort(order, 1) == np.arange(self.D)).all(), 'G2: not a permutation'
        if not whole_list:
            fixed = self.fam < 0
            assert (order[fixed] == np.nonzero(fixed)[1]).all(), 'G1: a non-family candidate moved'

    def oracle(self):
        """Perfect within-family reorder: the gold takes its family's first position."""
        r = self.gold_rank.copy()
        fam_first = np.array([np.nonzero(self.fam[i] == self.gold_fam[i])[0].min() + 1 if self.gold_fam[i] >= 0
                              else 0 for i in range(self.M)])
        better = (self.gold_fam >= 0) & (fam_first < r)
        r[better] = fam_first[better]
        return r

    def metrics(self, r, top1, sel, r0):
        n = max(int(sel.sum()), 1)
        sib = sel & (r > 1) & (self.gold_fam >= 0) & (self.fam[np.arange(self.M), top1] == self.gold_fam)
        return {'n': int(sel.sum()), 'acc@1': float((r[sel] == 1).mean()) if sel.any() else 0.0,
                'mrr': float((1.0 / np.maximum(r[sel], 1)).mean()) if sel.any() else 0.0,
                'acc@64': float((r[sel] <= 64).mean()) if sel.any() else 0.0,
                'sib_conf': float(sib.sum() / n),
                'fixed': int((sel & (r0 > 1) & (r == 1)).sum()), 'created': int((sel & (r0 == 1) & (r > 1)).sum()),
                'umbrella_demoted': int((sel & self.gold_umbrella & (r0 == 1) & (r > 1)).sum())}


def entity_stats(L, r0, r1, sel, n_perm=10000, n_boot=2000, seed=0):
    """Δacc@1 (r1 - r0) over mentions in sel: entity-level bootstrap 95% CI and sign-flip permutation p."""
    if not sel.any():
        return 0.0, 0.0, 0.0, 1.0
    d = (r1[sel] == 1).astype(float) - (r0[sel] == 1).astype(float)
    _, inv = np.unique(L.entity[sel], return_inverse=True)
    de, ne = np.bincount(inv, weights=d), np.bincount(inv)
    rng = np.random.default_rng(seed)
    obs = de.sum() / ne.sum()
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(de))) * de).sum(1) / ne.sum()
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    idx = rng.integers(0, len(de), size=(n_boot, len(de)))
    lo, hi = np.percentile(de[idx].sum(1) / ne[idx].sum(1), [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(p)


# ---------------------------------------------------------------- rows, tuning, test
ROWS = {  # name: (evidence, free knobs, whole_list)
    'recency': ('rec', ('eta',), False),
    'snapshot-year': ('snap', ('eta',), False),
    'placebo': ('plac', ('eta', 'lam'), False),
    'title-matcher': ('w_any', ('eta',), True),
    'lam-only': ('w', ('lam',), False),
    'eta-only': ('w', ('eta',), False),
    'B': ('w', ('eta', 'lam'), False),
}


def grid(free):
    for eta in (ETAS if 'eta' in free else (0.0,)):
        for lam in (LAMS if 'lam' in free else (0.0,)):
            yield eta, lam


def select(L, r0, mask, ev, free, whole, c, max_cc_drop=0.005):
    """Pre-registered rule on the mentions in `mask`: max new acc@1 subject to changeable-continual
    Δacc@1 >= -0.5 pt; ties -> smaller eta, then smaller lam; lam-guard on the gold-not-latest subset."""
    new, cc = mask & L.new, mask & L.cont & L.changeable
    sub = mask & L.new & (L.gold_fam >= 0) & ~L.gold_latest
    base_cc, base_sub = (r0[cc] == 1).mean() if cc.any() else 0, (r0[sub] == 1).mean() if sub.any() else 0
    best = None
    for eta, lam in grid(free):
        if lam > 0 and 'lam' in free:  # lam-guard: lam alone must not hurt the gold-not-latest subset
            rg, _ = L.ranks(L.order(ev, lam, 0.0, c))
            if sub.any() and (rg[sub] == 1).mean() < base_sub:
                continue
        r, _ = L.ranks(L.order(ev, lam, eta, c, whole))
        if cc.any() and (r[cc] == 1).mean() - base_cc < -max_cc_drop:
            continue
        key = ((r[new] == 1).mean(), -eta, -lam)
        if best is None or key > best[0]:
            best = (key, {'eta': eta, 'lam': lam})
    return best[1] if best else {'eta': 0.0, 'lam': 0.0}


def fold_ranks(L, knobs, name):
    """Ranks when each fold uses the knobs chosen on the other folds."""
    ev, free, whole = ROWS[name]
    r = np.zeros(L.M, dtype=np.int64)
    top1 = np.zeros(L.M, dtype=np.int64)
    for k in range(5):
        cfg = knobs['rows'][name][str(k)]
        o = L.order(ev, cfg['lam'], cfg['eta'], knobs['c'], whole)
        L.check(o, whole)
        rk, tk = L.ranks(o)
        r[L.fold == k], top1[L.fold == k] = rk[L.fold == k], tk[L.fold == k]
    return r, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prep', 'diagnose', 'tune', 'test'])
    ap.add_argument('prefix')
    ap.add_argument('--jsonl')
    ap.add_argument('--depth', type=int, default=64)
    ap.add_argument('--knobs')
    args = ap.parse_args()
    if args.cmd == 'prep':
        return prep(args.prefix, args.jsonl, args.depth)

    L = Lists(args.prefix, args.depth)
    o0 = L.order()
    L.check(o0)
    assert (o0 == np.arange(L.D)).all(), 'G4: lam = eta = 0 must reproduce the frozen list'
    r0, t0 = L.ranks(o0)
    assert (r0[L.gold >= 0] == L.gold_rank[L.gold >= 0]).all()

    if args.cmd == 'diagnose':
        ro = L.oracle()
        top_fam = L.fam[np.arange(L.M), t0]
        for name, sel in [('new: all', L.new)] + [(f'new: {t[7:]}', L.new & (L.type == i)) for i, t in
                                                    enumerate(TYPES)] + [('continual', L.cont)]:
            n = max(sel.sum(), 1)
            b = {'correct@1': sel & (r0 == 1),
                 'edition confusion @1': sel & (r0 > 1) & (r0 <= L.D) & (L.gold_fam >= 0) & (top_fam == L.gold_fam),
                 'miss, sibling in list': sel & (r0 > L.D) & L.miss_sib,
                 'miss, no sibling': sel & (r0 > L.D) & ~L.miss_sib}
            b['other in-list error'] = sel & (r0 > 1) & (r0 <= L.D) & ~b['edition confusion @1']
            print(f'{name} (n={sel.sum()}): ' + '; '.join(f'{k} {v.sum()} ({v.sum() / n:.1%})' for k, v in b.items())
                  + f' | oracle acc@1 {(ro[sel] == 1).mean():.4f}')
        return

    if args.cmd == 'tune':
        c = float(np.median(L.scores[:, 0] - L.scores[:, 1]))
        knobs = {'tuned_on': args.prefix, 'depth': args.depth, 'c': c, 'rows': {}, 'all_val': {}}
        for name, (ev, free, whole) in ROWS.items():
            knobs['rows'][name] = {str(k): select(L, r0, L.fold != k, ev, free, whole, c) for k in range(5)}
            knobs['all_val'][name] = select(L, r0, np.ones(L.M, bool), ev, free, whole, c)
            r, t = fold_ranks(L, knobs, name)
            print(f'{name:14s} folds {[tuple(v.values()) for v in knobs["rows"][name].values()]} '
                  f'| cross-fitted val: {json.dumps(L.metrics(r, t, L.new, r0))}')
        print('frozen', json.dumps(L.metrics(r0, t0, L.new, r0)), '| oracle new acc@1',
              float((L.oracle()[L.new] == 1).mean()))
        out = os.path.join(os.path.dirname(args.prefix), f'knobs_d{args.depth}.json')
        json.dump(knobs, open(out, 'w'), indent=1)
        print('saved', out)
        return

    knobs = json.load(open(args.knobs))
    assert knobs['tuned_on'] != args.prefix and knobs['depth'] == args.depth, 'knobs must come from validation'
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print(f'test on {args.prefix}, knobs {args.knobs} (tuned on {knobs["tuned_on"]}), git {git}')
    cc = L.cont & L.changeable
    sub = L.new & (L.gold_fam >= 0) & ~L.gold_latest
    res = {'frozen': L.metrics(r0, t0, L.new, r0), 'oracle new acc@1': float((L.oracle()[L.new] == 1).mean())}
    for name in ROWS:
        r, t = fold_ranks(L, knobs, name)
        res[name] = {'new': L.metrics(r, t, L.new, r0), 'delta_new': entity_stats(L, r0, r, L.new),
                     'gold_not_latest': entity_stats(L, r0, r, sub), 'n_gold_not_latest': int(sub.sum()),
                     'dated_non_snap': entity_stats(L, r0, r, L.new & L.dated_non_snap),
                     'n_dated_non_snap': int((L.new & L.dated_non_snap).sum()),
                     'changeable_continual': entity_stats(L, r0, r, cc), 'n_changeable_continual': int(cc.sum()),
                     'continual': L.metrics(r, t, L.cont, r0)}
        print(name, json.dumps(res[name]))
    out = f'{args.prefix}.results_d{args.depth}.json'
    assert not os.path.exists(out), f'{out} exists: one test pass per row (delete it only to rerun deliberately)'
    json.dump({'git': git, 'knobs': knobs, 'results': res}, open(out, 'w'), indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
