# CLAUDE.md

Research project on **emerging-entity linking** under temporal drift, evaluated on TempEL. Stage: proposal with early results (reproduction, diagnosis, the pre-registered test and the TIGER/BLINK re-scoring are done; the report and the team guide are written; the hand audit and the PPT are pending).

## Files
- `PROPOSAL.md`: **current working design and results**: a training-free within-family reorder on TempEL's released 2019 model and 2021 encodings. Edit this file when the design or results change.
- `report/report.tex`: IEEE-style report (single file, inline bibliography; compile on Overleaf, no LaTeX on the laptop). `README.md`: plain-language guide for the three teammates, with the numbers and expected questions. Both copy their numbers from `PROPOSAL.md`; change a number there first.
- `docs/ideas/family-reranker.md`: final method design (v3) and the pre-registration. `docs/ideas/review-round1.md`: the review record.
- `code/`: `download.sh`, `encode_mentions.py`, `retrieve.py`, `families.py`, `rerank.py` (prep / diagnose / tune / test), `t3_depth256.py`, `audit_sheet.py`, `kaggle_c1.sh`, `demo.py` (shows one mention's stored top-10 before and after the reorder, for a live demo), `tiger_b.py` (the reorder on the Kaggle dumps of TIGER's and BLINK's lists; one pass, do not rerun). Run from the repo root with `.venv/bin/python` and `PYTHONPATH=code`.
- `data/` (gitignored): TempEL downloads in `data/tempel/`, outputs in `data/enc/`, the audit sheet in `data/audit/`, Kaggle outputs and `tiger_b.json` in `data/kaggle/`, TIGER's released files in `data/tiger/`.
- `MYexplaination.md`: what was pitched to the professor. **Not a commitment.**
- `emerging-entity-linking-research-ideation.md`: long ideation doc. **Not a commitment.** Reuse its evaluation and falsification sections (§13–16).
- `Papers/`: TempEL (NeurIPS'22), TIGER, RAED, and `TempEL-Dx` (the user's own protocol/dataset design). `new entity.pdf` is a second copy of RAED.
- Read PDFs with `pdftotext -layout <pdf> <out>`. Use `grep -a`, because the TIGER text has binary glyphs.

Neither draft is a spec. Judge ideas on evidence from the papers, and correct errors in either draft instead of repeating them.

## Facts that are easy to get wrong
- New entities **degrade** more (17.9 vs 3.1 acc@64), but in TempEL they score **higher** in absolute acc@64 (0.910 vs 0.785 in the 2013 cell).
- About 90% of new entities are recurring events (~68%) or page extracts (~22%). They are *new pages from existing families*, **not** continual entities.
- COVID-19 in 2021: 24% of new entities, 30% of mentions, ~14 extra points lost. This is the novel-knowledge population.
- TIGER shapes *entity* embeddings with graphs during **training** (Wikidata5M from July 2019, plus a description-kNN graph). It does not enrich the mention. Text-only inference is claimed by the drafts; confirm it in the code.
- On new entities, CYCLE ≈ TIGER (R@1 at gap 3: .222 vs .217). The CYCLE paper is not in `Papers/`.
- RAED's index (Jan 2023 dump) is not temporally sound. Emerging acc@64 is 57.0/47.4/53.5 vs a gold ceiling of 71.0/64.8/72.3.
- TIGER/CYCLE's "new entity" test set **is** pooled: 46,765 continual + 1,450 new mentions (verified in TIGER's code and data). Their pool has 10,373 entities; gap 3 is the single pair 2019→2022. The public CYCLE repo cannot run.
- TempEL's new entities carry a **recency artifact**: 81% of dated new gold titles are 2020 editions, so "prefer the 2020 edition" ties our method on new-entity acc@1 (+3.2 vs +3.6) and beats it on acc@64 at depth 256 (0.764 vs 0.751). Our method differs on continual entities (+2.5 vs −0.3).
- Our re-run of TIGER's public code (one seed, verified, `data/kaggle/`): trained on new, 2019→2022, pooled R@1 is BLINK 0.193 / TIGER 0.199 (published .142 / .217); new-only 0.343 / 0.349; continual-only 0.189 / 0.195. TIGER − BLINK is +0.5 to +0.7 points pooled and not significant on new-only in any year. Never write that TIGER's paper is wrong.
- On TIGER's 10,373-entity pool our method gains +0.4 to +2.2 on new R@1 and about +2 on continual R@1, but the **title-matcher control beats it** there (+5 to +12 new). On TempEL's full pool it was the reverse (+1.0 vs +3.6).
- The test split was scored **once**, after git `12fd858`. Do not rerun `rerank.py test` or `t3_depth256.py`; anything new on test is exploratory and must be labelled so.
- Re-ordering the top-64 cannot change acc@64; only the depth-256 table can.

## Writing rules
- Quote numbers only from papers, and name the source. Label each claim as verified, inferred, or reported secondhand.
- Make no performance predictions.
- Follow the strict $\le t$ temporal protocol. Report any LLM trained on post-$t$ data as a contaminated oracle.
- Keep sibling-confusion rate and continual-entity non-regression in every evaluation.
