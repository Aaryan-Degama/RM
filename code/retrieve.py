"""Exact top-K retrieval over TempEL's released entity encodings, without loading them.

*_tensors.pt is a plain pickle (protocol 5) of {'tensors': float32 ndarray}
(TempEL s05_wikipedia_encode_utils.py:114-121). numpy pickles it as
_frombuffer(bytearray8, dtype, shape, order), so the raw rows sit at a fixed
offset and can be np.memmap'ed. layout() checks the header and the pickle tail
(dtype '<f4', shape, order 'C') before trusting that offset.

Scores are raw dot products, exact search (TempEL uses faiss IndexFlatIP).
Also computes each mention's exact gold rank over the whole pool.

  python code/retrieve.py data/enc/test2021 --k 256
writes OUT.top256.npz: ids, scores (M,K); gold, gold_rank (1-based, 0 = gold not in pool)
"""
import argparse
import pickle
import pickletools
import json
import os

import numpy as np
import torch

ENC = 'data/tempel/2019/2021-01-01T00:00:00Z_tensors.pt'
MAP = 'data/tempel/2019/2021-01-01T00:00:00Z_label_id_to_wikidata_qid_mappings.pt'


def layout(path):
    """Return (offset, rows, dim) of the float32 matrix inside the pickle."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(256)
        p = head.index(b'\x96')  # BYTEARRAY8 opcode
        nbytes = int.from_bytes(head[p + 1:p + 9], 'little')
        offset = p + 9
        f.seek(offset + nbytes)
        tail = f.read()
    assert offset + nbytes + len(tail) == size
    ops = [(op.name, a) for op, a, _ in pickletools.genops(tail)]
    args = [a for _, a in ops]
    assert '<' in args and 'f4' in args and 'C' in args, ops  # little-endian float32, C order
    ints = [a for name, a in ops if name.startswith(('BININT', 'LONG', 'INT'))]
    rows, dim = ints[-2], ints[-1]  # the shape tuple is the last pair of ints in the tail
    assert rows * dim * 4 == nbytes, (rows, dim, nbytes)
    return offset, rows, dim


def label_qids(map_path=MAP):
    """label_id -> numeric QID array (int64), cached next to the mapping."""
    cache = map_path.replace('.pt', '.qid.npy')
    if os.path.exists(cache):
        return np.load(cache)
    d = pickle.load(open(map_path, 'rb'))
    m = d['label_id_to_wikidata_qid']
    q = np.zeros(len(m), dtype=np.int64)
    for i, s in m.items():
        q[i] = int(s[1:])
    assert len(m) == d['embeddings_tensor_shape'][0]
    np.save(cache, q)
    return q


def entity_norms(enc=ENC, chunk=65536):
    """||g||^2 for every entity (float32), one sequential pass, cached next to the table."""
    cache = enc.replace('.pt', '.norms.npy')
    if os.path.exists(cache):
        return np.load(cache, mmap_mode='r')
    offset, rows, dim = layout(enc)
    E = np.memmap(enc, dtype='<f4', mode='r', offset=offset, shape=(rows, dim))
    n = np.empty(rows, dtype=np.float32)
    for s in range(0, rows, chunk):
        x = np.asarray(E[s:s + chunk], dtype=np.float64)
        n[s:s + chunk] = np.einsum('ij,ij->i', x, x)
    np.save(cache, n)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', help='output prefix of encode_mentions.py')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--enc', default=ENC)
    ap.add_argument('--chunk', type=int, default=32768)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    offset, rows, dim = layout(args.enc)
    E = np.memmap(args.enc, dtype='<f4', mode='r', offset=offset, shape=(rows, dim))
    qids = label_qids()
    assert len(qids) == rows
    qid2label = {int(q): i for i, q in enumerate(qids)}
    meta = [json.loads(l) for l in open(args.prefix + '.meta.jsonl', encoding='utf-8')]
    gold = np.array([qid2label.get(int(m['target_qid'][1:]), -1) for m in meta])
    del qid2label
    print(f'pool {rows}x{dim}; {len(meta)} mentions, {(gold < 0).sum()} gold QIDs missing from pool')

    Q = torch.from_numpy(np.load(args.prefix + '.npy'))
    g = torch.from_numpy(np.ascontiguousarray(E[np.maximum(gold, 0)])).float()
    gold_score = (Q * g).sum(1)
    above = torch.zeros(len(Q), dtype=torch.long)
    top_s = torch.full((len(Q), args.k), -float('inf'))
    top_i = torch.zeros((len(Q), args.k), dtype=torch.long)
    gold_t = torch.from_numpy(gold)
    for n, s in enumerate(range(0, rows, args.chunk)):
        S = Q @ torch.from_numpy(np.array(E[s:s + args.chunk])).T
        above += (S > gold_score[:, None]).sum(1)
        # the GEMM score of the gold itself can exceed its elementwise score by ~1e-5:
        # never let the gold count as ranked above itself
        r = ((gold_t >= s) & (gold_t < s + S.shape[1])).nonzero().squeeze(1)
        above[r] -= (S[r, gold_t[r] - s] > gold_score[r]).long()
        cs, ci = S.topk(min(args.k, S.shape[1]), dim=1)
        top_s, idx = torch.cat([top_s, cs], 1).topk(args.k, dim=1)
        top_i = torch.cat([top_i, ci + s], 1).gather(1, idx)
        if n % 20 == 0:
            print(f'{min(s + args.chunk, rows)}/{rows}', flush=True)

    rank = np.where(gold >= 0, above.numpy() + 1, 0)
    # inside the list, its GEMM-sorted position is authoritative; report any disagreement
    hit = top_i.numpy() == gold[:, None]
    inlist = hit.any(1)
    pos = hit.argmax(1) + 1
    print(f'{(rank[inlist] != pos[inlist]).sum()} of {inlist.sum()} in-list gold ranks disagree with list position '
          f'(float near-ties); using list position')
    rank[inlist] = pos[inlist]
    out = f'{args.prefix}.top{args.k}.npz'
    np.savez(out, ids=top_i.numpy().astype(np.int32), scores=top_s.numpy(), gold=gold,
             gold_rank=rank, gold_score=gold_score.numpy())
    new = np.array([m['category'] == 'new_entities' for m in meta])
    for name, sel in (('new', new & (gold >= 0)), ('shared', ~new & (gold >= 0))):
        r = rank[sel]
        print(name, len(r), ' '.join(f'acc@{k}={np.mean(r <= k):.4f}' for k in (1, 8, 64, 256)))
    print('saved', out)


if __name__ == '__main__':
    main()
