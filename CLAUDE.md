# CLAUDE.md

Research project on **emerging-entity linking** under temporal drift, evaluated on TempEL. Stage: proposal / design. **No code yet.**

## Files
- `PROPOSAL.md`: **current working design**. It covers the motivation, problem statement, the family–residual model, the architecture, and how each TempEL problem maps to a component. Edit this file when the design changes.
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
- TIGER/CYCLE's "new entity" test set may be pooled (~97% continual). This is inferred in TempEL-Dx and unverified.

## Writing rules
- Quote numbers only from papers, and name the source. Label each claim as verified, inferred, or reported secondhand.
- Make no performance predictions.
- Follow the strict $\le t$ temporal protocol. Report any LLM trained on post-$t$ data as a contaminated oracle.
- Keep sibling-confusion rate and continual-entity non-regression in every evaluation.
