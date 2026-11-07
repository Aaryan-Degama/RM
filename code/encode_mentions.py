"""Encode TempEL mentions with the released 2019 context encoder (CPU).

Input construction is copied verbatim from TempEL
src/models/biencoder/data_process.py:38-78 (get_context_representation) and
src/models/biencoder/biencoder.py:153 (to_bert_input). The embedding is the
last-layer [CLS] vector (src/models/utils/ranker_base.py, add_linear=false).

Selects every new-entity mention plus a seeded sample of continual ("shared")
mentions. Writes OUT.npy (float32, M x 1024, row order = OUT.meta.jsonl) and
resumes from OUT.progress if interrupted.

  python code/encode_mentions.py data/tempel/.../2021-01-01T00:00:00Z_test.jsonl \
      data/enc/test2021 --shared 5000
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import BertConfig, BertModel

ENT_START_TAG, ENT_END_TAG = '[unused0]', '[unused1]'
NULL_IDX = 0


def get_context_representation(sample, vocab, max_seq_length,
                               ent_start_token=ENT_START_TAG, ent_end_token=ENT_END_TAG):
    # verbatim from TempEL data_process.py:38-78; tokenizer.convert_tokens_to_ids
    # in pytorch_transformers is vocab.get(token, vocab['[UNK]'])
    mention_tokens = []
    if sample['mention_bert'] and len(sample['mention_bert']) > 0:
        mention_tokens = sample['mention_bert']
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample['context_left_bert']
    context_right = sample['context_right_bert']

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (context_left[-left_quota:] + mention_tokens + context_right[:right_quota])

    context_tokens = ['[CLS]'] + context_tokens + ['[SEP]']
    input_ids = [vocab.get(t, vocab['[UNK]']) for t in context_tokens]
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length
    return {'tokens': context_tokens, 'mention_tokens': mention_tokens, 'ids': input_ids}


def to_bert_input(token_idx, null_idx=NULL_IDX):
    # verbatim from TempEL biencoder.py:153
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


def load_context_encoder(model_path):
    sd = torch.load(model_path, map_location='cpu', mmap=True, weights_only=True)
    pre = 'context_encoder.bert_model.'
    sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    vocab_size, hidden = sd['embeddings.word_embeddings.weight'].shape
    assert hidden == 1024, hidden  # bert-large-uncased (TempEL footnote 6)
    cfg = BertConfig(vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=24,
                     num_attention_heads=16, intermediate_size=4096, hidden_act='gelu',
                     max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12)
    model = BertModel(cfg, add_pooling_layer=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert all('position_ids' in k for k in missing), missing
    assert all(k.startswith('pooler.') or 'position_ids' in k for k in unexpected), unexpected
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl')
    ap.add_argument('out')
    ap.add_argument('--shared', type=int, default=5000, help='continual mentions to sample')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='encode only the first N selected (smoke test)')
    ap.add_argument('--model', default='data/tempel/2019/model_epoch_09.bin')
    ap.add_argument('--vocab', default='data/tempel/2019/vocab.txt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    vocab = {t.rstrip('\n'): i for i, t in enumerate(open(args.vocab, encoding='utf-8'))}
    rows = [json.loads(l) for l in open(args.jsonl, encoding='utf-8')]
    new = [i for i, r in enumerate(rows) if r['category'] == 'new_entities']
    shared = [i for i, r in enumerate(rows) if r['category'] == 'shared']
    sel = sorted(new + random.Random(args.seed).sample(shared, min(args.shared, len(shared))))
    if args.limit:
        sel = sel[:args.limit]
    print(f'{len(rows)} mentions: {len(new)} new, {len(shared)} shared; encoding {len(sel)}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    prog = args.out + '.progress'
    if os.path.exists(prog):  # resuming: the selection must be identical, row for row
        old = [json.loads(l)['line'] for l in open(args.out + '.meta.jsonl', encoding='utf-8')]
        assert old == sel, 'resume with a different selection; delete the outputs or change OUT'
    with open(args.out + '.meta.jsonl', 'w', encoding='utf-8') as f:
        for i in sel:
            r = rows[i]
            ctx = ' '.join(r['context_left_bert'] + r['mention_bert'] + r['context_right_bert']).replace(' ##', '')
            f.write(json.dumps({'line': i, 'category': r['category'], 'target_qid': r['target_qid'],
                                'target_title': r['target_title'], 'mention': r['mention'],
                                'ctx': ctx}) + '\n')
    ids = torch.tensor([get_context_representation(rows[i], vocab, 128)['ids'] for i in sel])
    del rows

    done = int(open(prog).read()) if os.path.exists(prog) else 0
    mode = 'r+' if done else 'w+'
    out = np.lib.format.open_memmap(args.out + '.npy', mode=mode, dtype=np.float32, shape=(len(sel), 1024))
    assert out.shape == (len(sel), 1024), f'resume with a different selection: {out.shape} vs {len(sel)}'
    torch.set_num_threads(args.threads)
    model = load_context_encoder(args.model)
    with torch.inference_mode():
        for s in range(done, len(sel), args.batch):
            tok, seg, mask = to_bert_input(ids[s:s + args.batch])
            out[s:s + len(tok)] = model(tok, attention_mask=mask, token_type_ids=seg).last_hidden_state[:, 0].numpy()
            if (s // args.batch) % 20 == 0 or s + args.batch >= len(sel):
                out.flush()
                open(prog + '.tmp', 'w').write(str(min(s + args.batch, len(sel))))
                os.replace(prog + '.tmp', prog)
                print(f'{min(s + args.batch, len(sel))}/{len(sel)}', flush=True)
    out.flush()
    print('done', args.out + '.npy')


if __name__ == '__main__':
    main()
