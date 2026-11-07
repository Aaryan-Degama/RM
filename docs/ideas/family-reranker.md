# Edition confusion in emerging-entity linking — final design (v3)

Status: **final**, 2026-11-06, after two rounds of adversarial review (NLP area chair, evaluation
auditor, method theorist, research engineer, hostile professor) and an overall ruling; full record
in `review-round1.md`. Labels: *verified* (measured by us), *reported* (from a paper), *inferred*.
No performance predictions.

## Claim
On TempEL 2019→2021, using the authors' own released model and pool:
- **C1** — first new-only scores for TIGER/BLINK: their "new entity" test set is 97% continual (verified).
- **C2** — edition confusion (a sibling edition or the family's umbrella page ranked above the gold)
  is a measured, sizable share of new-entity errors, bounded by an oracle and a hand audit.
- **C3** — a training-free within-family reorder that cannot move non-family candidates, judged by
  one pre-registered test against recency, snapshot-year, placebo and title-only controls.
Never claimed: "state of the art", "beats TIGER/CYCLE", "TIGER/CYCLE are wrong", "better than TempEL
at acc@64 (depth 64)", anything about COVID/novel entities, numeric comparison with RAED.

## Evidence so far (verified)
| | value |
|---|---|
| G0 — frozen TempEL 2019 model, full 2021 pool (6,180,587), test acc@64 new | **0.7345** vs 0.734 reported (TempEL Table 2) — pass |
| G0 — continual acc@64 (seeded 5,000 sample) | 0.7736 vs 0.767 reported — inside the sample's CI |
| Frozen test acc@1 (never published by TempEL) | new 0.1738, continual 0.2052 (sample) |
| Validation oracle: perfect within-family reorder, new (1,231) | 0.164 → **0.268 (+10.4 pt)**: 86 sibling + 42 umbrella rank-1 errors of 714 in-list errors |
| Validation oracle, continual (5,000) | 0.207 → 0.231 (+2.4 pt) |
| Gold has a title sibling in the pool (2021 test) | full pool: new 49.1%, continual 24.3%; TIGER pool: 17.4% / 18.6% |
| **Recency artifact** | continual gold titles never carry a year ≥ 2013; "2020" is in the 128-token window of 49% of new vs 1.5% of continual mentions; 81% of dated new golds are 2020 editions; "pick the latest edition" finds the gold for 76% of new family mentions in the full pool |

## Algorithm (v3)
Per mention: frozen top-K list (K = 64 primary; 256 secondary), scores s, entity norms
n(e) = ‖g_e‖² (one sequential pass over the pool), as-of-t titles (100% coverage asserted), the
128-token encoder window W containing the mention span M.

1. `tau(title)` → template T, slot type κ ∈ {year, season, ordinal, roman, numword}, slot value v
   (name-internal numbers like "COVID-19" are not slots; strict Roman numerals only).
2. **Families**: group list candidates by (T, κ); add the **umbrella** page (title = T without the
   slot); union-find merge of families sharing a member; keep |F| ≥ 2; assert a partition.
3. **Slot evidence** w(e) = max over occurrences of q·ρ: q = 1 exact typed match, ½ a bare year that
   is a season endpoint; ρ = 1 inside M, ½ elsewhere in W; non-year slots count only inside M;
   the umbrella has w = 0.
4. **Score** z(e) = s(e) − (λ/2)·n(e) + η·c·w(e). (λ = 1 ranks by −‖f − g‖².) c = validation median of
   s₁ − s₂ (label-free), per base model.
5. **Ordering (B)**: each family refills exactly the positions it already holds, sorted by
   (−z, original rank).

Guarantees (asserted on every configuration): G1 non-family candidates never move; G2 acc@K at the
list depth unchanged; G3 Δacc@1 = sibling fixes − sibling creations (sibling = same merged family,
umbrella included); G4 λ = η = 0 reproduces the frozen list.

Tuned knobs: η ∈ {0, .5, 1, 2, 4, ∞}, λ ∈ {0, .25, .5, 1}. Fixed: c, q, ρ, window.
λ-guard: λ := 0 if λ-only hurts the validation gold-not-latest subset (the norm may proxy page age — inferred).

## Controls (same B ordering unless stated)
recency prior w = [v is the largest slot value in F]; snapshot-year prior w = [2020 ∈ v]; placebo
(W shuffled among mentions whose rank-1 candidates share a template); λ-only; η-only; title-matcher
and BM25 (whole-list sort, no families). Cut: random families.

## Tables for Nov 20
- **T1 primary (TempEL test, depth 64)**: frozen, oracle, recency, snapshot-year, placebo,
  title-matcher, BM25, λ-only, η-only, **B (cross-fitted)**, B (one config on all validation).
  Columns: new acc@1, Δ + CI + p, gold-not-latest subset (n first), dated non-2020 golds (n = 129),
  MRR, sibling-confusion, changeable-continual Δ and lower bound, continual acc@1, umbrella demotions.
- **T2 C2 (frozen)**: new by slice (all / recurrent / COVID / other) + continual × {correct, edition
  confusion at rank 1, other in-list error, miss with sibling in top-64, miss without}; hand-audit
  agreement; G0 vs TempEL Table 2.
- **T3 depth 256** (same knobs): acc@64 frozen vs B, acc@256, oracle.
- **T4 C1 — TIGER's protocol, full grid** (student's decision): BLINK (baseline.yaml) and TIGER
  (paper.yaml) × {new_entities, continual_entities} × train years 2019–2022 = 16 trainings, tested
  forward to 2022; published vs re-run; R@1, R@64 × pooled / new / continual × gaps 0–3; TIGER with
  titles in the eval pool; ±B on the 2019 models with TempEL's knobs unchanged (c from TIGER's own
  lists); no Π_W/α on TIGER's pool (every pool entity is a test gold). CYCLE quoted only.
- **T5 exploratory** (after the test run, labelled): edition subspace Π_W (r = 8 fixed, α ∈ {0, .5,
  1, 2}, fit on families with no val/test gold); BLINK cross-encoder vs cross-encoder + B (gate:
  Tue Nov 11 12:00 with T1/C1/audit done; no numbers by Wed 12:00 → future work).

## Pre-registration (commit before any test run; G0 already exempt)
1. Data: TempEL 2021 test, released 2019 bi-encoder, full pool, top-64; 1,450 new + 46,765 continual
   mentions (fallback: the seeded 5,000, labelled underpowered).
2. Primary: Δ new-entity acc@1 = B (cross-fitted) − frozen; entity-level sign-flip permutation,
   10k permutations, two-sided α = 0.05.
3. Non-inferiority: continual mentions whose rank-1 lies in an in-list family — 95% entity-bootstrap
   lower bound of Δacc@1 ≥ −1.0 pt. δ is never widened.
4. Cross-fitting: 5 folds grouped by gold template (val and test share every entity); fold k uses
   knobs chosen on the other folds' validation: max new acc@1 s.t. changeable-continual Δ ≥ −0.5 pt;
   ties → smaller η, then smaller λ. Grid, constants and λ-guard as above, validation only.
5. Secondary (Holm): B vs each control on all new, gold-not-latest, dated non-2020.
6. Always reported: sibling-confusion, created-error types, T3, T4.
7. One test pass per row; anything later is exploratory.

## Decision rules
- Oracle tiny (< 1.5 pt on validation): lead with C2/C1 — **not triggered** (+10.4 pt).
- B only ties the recency prior on validation: primary contrast unchanged; headline becomes the
  continual cost of recency vs B on the gold-not-latest subset; if B also fails there, C2 reports
  edition confusion as largely a recency artifact of TempEL's construction. Never add recency to B.
- G0 off by > 0.015: **not triggered** (0.0005).

## Schedule
Fri 7: families v3, norm pass, encoder-window extraction, encode resume fix; Kaggle grid (patch
dumps top-64 ids/scores); overnight: remaining 41,765 continual test mentions (`--chunk` ≤ 4096).
Sat 8: rerank v3; validation oracle/rows/λ-guard; commit pre-registration. Sun 9: C1 table, TIGER ±B,
audit sheet. Mon 10: single test run, C2, audit agreement. Tue 11: cross-encoder gate / Π_W.
Wed 12–Fri 14: write-up. Sat 15: buffer.
Never cut: G0; C1 new-only; C2 + oracle; pre-registered B; recency, snapshot-year, placebo,
title-matcher; both subsets; continual non-inferiority; audit ≥ 50.
Cut order: cross-encoder → Π_W → BM25 → TIGER grid beyond 2019 (student chose the full grid; it
slips last-but-not-least only if the never-cut list is at risk) → audit 50 → T3 → full continual set.
