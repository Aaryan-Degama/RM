# Review round 1 — responses and revised design (v2)

Critics: NLP area chair (novelty), evaluation/leakage auditor, method/math theorist,
research engineer, hostile professor. 2026-11-06. Verified = checked by us or a critic on data.

## What round 1 established (accepted)
| # | Finding | Source | Status |
|---|---|---|---|
| 1 | Per-family PCA term removes the directions where siblings **differ**; at \|F\|=2 the discount is exactly a squared-norm penalty (checked to 1e-13); under within-family reorder the family mean term cancels | method, NLP, professor | **Accepted — (β, k) dropped** |
| 2 | `rerank.py` implements the global sort (A); with an uncentred shared term whole families sink by ~40 points, so A is degenerate; D equals B at R@1 | method, engineer | **Accepted — B only; A kept as one negative ablation row; D dropped** |
| 3 | **Recency artifact**: continual gold titles never carry a year ≥ 2013 (max 2012); "pick the latest edition, ignore context" finds the gold for 89% of new family mentions in the gold-only pool and 76% in the full pool; "2020" is in 57% of new vs 2% of continual contexts | evaluation (verified on 2021 files) | **Accepted — mandatory controls + subset reporting** |
| 4 | `retrieve.py` gold counted itself (GEMM vs elementwise ±4.6e-5) for 60/175 in-list golds | engineer (verified) | **Fixed**; smoke re-run: 0/175 disagreements |
| 5 | Validation and test share every target entity (373/373 new, 10,000/10,000 continual) | evaluation (verified) | **Accepted — template-grouped cross-fitting** |
| 6 | Titles hard-coded to page_info 2021 (future titles for 2019/2020; 2022 new entities missing → silently no family) | evaluation | **Accepted — per (year, split) titles, assert 100% coverage** |
| 7 | Slot parsing loose ("CIVIL" as Roman numeral, "23rd" vs dates, seasons keyed to first year, slot words like "first"/"v" in normal text); `ctx` is wider (median 183 words) than the encoder window | method, engineer | **Accepted — typed slots, encoder window only** |
| 8 | Umbrella pages ("FIFA World Cup") have no slot, never join the family, sometimes rank first | method (2/200 smoke) | **Accepted — umbrella joins F with empty slot** |
| 9 | McNemar counts mentions (median 2, max 10 per new entity) → optimistic p | evaluation | **Accepted — entity-level sign-flip permutation; Holm** |
| 10 | Doc vs code objective conflict (pooled R@1 vs new acc@1 ≤0.2 pt continual drop) | all five | **Needs the student's decision (see below)** |
| 11 | TIGER comparison: our reranker reads titles TIGER's eval pool never encoded; "+ours" partly repairs TIGER's train/eval title mismatch | evaluation, NLP, professor | **Accepted — TIGER-with-titles vs TIGER-with-titles+ours** |
| 12 | Missing baselines: BLINK cross-encoder (trained on the May-2019 Wikipedia dump → temporally clean for 2020–22, reads titles, no training); BM25/title overlap; prior work Event Linking (Yu+ EACL'23) already names "repeating events" | NLP | **Partly accepted (scope, see below)** |
| 13 | Sibling-confusion measured with our own τ is circular | evaluation | **Accepted — add a 100-error hand audit as the independent label** |
| 14 | Smoke B ceiling: 6 of 120 in-list misses fixable by sibling reorder (+2 umbrella) — sibling headroom ≠ top-64 headroom | method (n=200, mostly continual) | **Accepted — oracle ceiling is step 1, before tuning** |

## Revised algorithm (v2)
Ordering: **B only** — each family F (title-template family **plus its umbrella page**) keeps
the positions it holds in the frozen list; members are re-sorted into them by z.

z(e) = s(e) − (λ/2)·(‖g_e‖² − mean_{F}‖g‖²) + α·⟨Π_W f, g_e − μ_F⟩ + η·c·Y(e)

- **λ ∈ [0,1] — dot product vs distance.** λ = 1 ranks the family by −‖f − g‖² (Euclidean /
  Gaussian likelihood); λ = 0 is the dot product. Reproduces the old β effect exactly at |F| = 2.
- **Π_W — a global "edition subspace".** Top-r eigenvectors of the within-family scatter
  S_W = Σ_F Σ_{e∈F} (g_e − μ_F)(g_e − μ_F)ᵀ, estimated **once** from ≤ 50k as-of-t pool entities
  in template families (n ≫ d, closed form). α is **signed**: α > 0 amplifies the mention's
  evidence along axes where editions differ; α < 0 tests the original "discount" hypothesis.
  r ∈ {8, 32, 128}. This is the defensible form of "prototype + residual": the family mean is
  the prototype, edition axes are learned from the KB, not from 2–10 siblings.
- **Y(e) — relative, typed slot evidence** inside the encoder's 128-token window:
  w(e) = max over occurrences o of slot(e) of q(o)·exp(−d(o)/ℓ), d = token distance to the
  mention span (0 inside); q = 1 exact typed match, ½ a bare year inside a season interval;
  non-year slots (ordinals, Roman numerals) count only inside the mention span.
  Y(e) = w(e) − max_{e′∈F∖e} w(e′): positive only for the single best-matching member, 0 on ties,
  negative for the umbrella when some edition matches.
- **c** — validation median top-1 − top-2 gap per base model (replaces σ_m).
- Knobs λ, α, r, η, ℓ; λ = α = η = 0 is the frozen baseline exactly. Every knob is ablated.

Properties (stated as safety properties, not contributions): G1 non-family candidates never
move; G3 Δacc@1 = sibling errors fixed − created (by τ and by hand audit); G4 nesting.

## Mandatory controls (every table)
recency prior ("latest edition in F"); title-number matcher without families (same knobs);
norm-only (λ only); context placebo (contexts shuffled within family); random families of the
same sizes; TIGER-with-titles ± ours. Report also on the subset where **the gold is not the
latest edition in F** and on continual mentions whose lists B can change.

## Decisions needed from the student (critics unanimous)
1. **Lead table**: TempEL full pool (49.1% vs 17.4% sibling coverage; titles already in the
   encoder; acc@1 never published by TempEL) with TIGER as a compatibility table — instead of
   TIGER's pool as the lead.
2. **Objective**: pre-registered new-entity acc@1 (TempEL 2021), cross-fitted by template folds,
   with a continual non-inferiority margin on changeable lists — instead of pooled R@1.
3. **Scope**: TIGER 2019-trained models only (4 trainings + title control); the 16-training grid
   only if ahead of schedule.

## Scope question for the PI
BLINK cross-encoder as a second, temporally clean base scorer (bi-encoder → cross-encoder → +B):
needs the 3.2 GB candidate token-id table + the BLINK checkpoint and ~1 h of Kaggle GPU for
~800k pairs. Strongest acc@1 story; adds risk to a 9-day, part-time schedule.

## Framing (accepted from NLP + professor)
"Edition confusion: the hidden failure mode of emerging-entity linking" — measurement under
temporal drift (first), a safe training-free within-family step that stacks on any scorer, gains
that survive recency, placebo and title-only controls. Forbidden: "state of the art", "beats
TIGER/CYCLE", "TIGER/CYCLE are wrong", "better than TempEL at acc@64 (depth 64)", "removes the
family-shared subspace", anything about COVID/novel entities, numeric comparison with RAED.

---

# Round 2 — critics' rebuttals of v2 (2026-11-06)

**Professor (panel).** v2 passes on design; Nov 20 will judge numbers. First objection still: the
recency artifact — the headline must be the "gold not latest edition" subset vs the recency prior,
with its size stated first; if B only ties recency, report that as a C2 finding. Second: knob creep
(5 knobs on ~373 entities) is the old over-scoping moved into tuning. λ: keep, call it a norm
correction. Π_W: better than per-family PCA but dressing for Nov 20; a signed α plus an r grid is a
fishing licence. Rulings: lead = TempEL full pool; objective = pre-registered new-entity acc@1 with a
continual margin; pooled R@1 reported, never optimised.

**NLP reviewer.** Resolved: PCA dropped, B only, framing, recency control. Unresolved: add one
BM25(context, title) row; decide lead/objective now (TempEL full pool, new-entity acc@1). Π_W is not
LDA (LDA suppresses these axes): α<0 = Nuisance Attribute Projection (Solomonoff+ 2005) / WCCN
(Hatch+ 2006), α>0 amplifies the PLDA within-class subspace (Ioffe 2006; Prince & Elder 2007);
operator not novel, the family-then-edition decomposition for EL is; motivate with linear time
directions (Gurnee & Tegmark 2023); S_W on the as-of-t pool is transductive over the KB (legal, say
so; strict variant: 2019 pool). λ is the known dot-vs-Euclidean norm bias, it is the norm-only
control. Fix ℓ and r a priori, tune ≤3 knobs, report sensitivity. Cross-encoder: gated stretch —
only if B oracle + controls on TempEL 2021 are done by Nov 11; new mentions + ~2k continual (~220k
pairs); BLINK's large cross-encoder sees only 32 context tokens, so B should be complementary.

**Method theorist.** Sound after two fixes. Under B, μ_F and mean‖g‖² are constant per family, so the
ordering score is z_B(e) = ⟨(I+αΠ_W)f, g_e⟩ − (λ/2)‖g_e‖² + ηcY(e) + const_F. G4 exact (sort by
(z, original position)). G1 exact only with disjoint families: merge families sharing a member
(union-find), assert a partition; define "one slotted member + umbrella" as a family. G3 exact with
sibling = same merged F incl. umbrella; new created-error type: gold umbrella demoted when Y<0 —
report that subset. Exclude umbrellas from S_W. Split-half stability on 49,943 entities / 8,943
year-prefixed families (≤10 members each): r=8 overlap 0.995 (eig gap 1.10, 42% within-family
variance), r=32 0.963, r=128 0.907 (poorly identified, near-uniform rescale). Ruling: **r = 8 fixed,
α ≥ 0** (grid {0, .5, 1, 2}; α<0 is the rejected discount hypothesis), tuned knobs **η, λ, α**; fixed
r, ℓ (e.g. 16 tokens), q, c. Essential rows: frozen, B-oracle, recency, η-only, λ-only, full,
placebo. Drop A, random families, r grid, α<0.

**Evaluation auditor.** Resolved on paper: cross-fitting, per-(year,split) titles, permutation+Holm,
hand audit, TIGER-with-titles ± ours, recency row + subset. Unresolved: objective/δ; knob count vs
~200 changeable new val mentions per fold; BM25 missing; non-inferiority sample small (~1,000
changeable continual of the 5,000 encoded). New: inside the 128-token window "2020" occurs in 711/1,450
new (49%) vs 686/46,765 continual (1.5%); 556/685 dated new golds (81%) are 2020 editions; the
within-family context placebo keeps that year distribution, so it is weak → add a **snapshot-year
prior** (Y = [slot ∋ t−1], same η) and report the **129 dated new mentions whose gold is not 2020**.
Π_W on TempEL's pool is acceptable transductive use only from a seeded uniform sample over all
template families; sensitivity row refits Π_W without families containing any val/test gold.
**Π_W on TIGER's pool is a leak** (all 10,373 entities are test golds) → fit outside that set or drop
α for TIGER. Pre-registration: primary = TempEL 2019→2021 full pool, depth 64, new-entity acc@1 over
all 1,450 mentions, B (cross-fitted) − frozen, entity sign-flip permutation, two-sided 0.05;
non-inferiority on changeable continual lists: 95% entity-bootstrap lower bound of Δacc@1 ≥ −1.0 pt
(≈10–12 mentions; ≈0.25 pt over all continual); if the validation half-width > 0.7 pt, encode more
continual mentions before test; never widen δ.

**Research engineer.** retrieve.py fix verified correct. encode_mentions resume assert partial: a
different selection with the same count slips through, and meta is rewritten before the assert →
compare saved `line` lists first. rerank.py/families.py/kaggle_c1.sh still v1 (A only, raw η, no TIGER
path, rank-file collision, pip fallback). Cost of full v2 ≈ 5 days vs ~6 available → no buffer.
Day plan: Thu G0 + typed τ + umbrella; Fri gather, B, oracle, recency, subset, C2, Kaggle start; Sat
Y (binary window/mention flag first), λ, c, cross-fitting, placebo/norm/title controls; Sun TIGER
adapter, C1, TIGER-with-titles ± B; Mon single test run, permutation+Holm, hand audit; Tue Π_W, ℓ or
random families if time; Wed–Fri write-up; Sat buffer. Cut order: cross-encoder → Π_W → ℓ decay →
random families → TIGER continual variant → TIGER+B (keep C1) → hand audit 50. Never cut: G0, oracle,
B with λ and η, recency + placebo controls, gold-not-latest subset, cross-fitting, C1, continual
non-inferiority. Cross-encoder GPU time fits (≈0.5–1.5 h for 220k pairs on T4) but needs ≈1 day of
code, TempEL-2021 candidate text extracted locally (≤300k rows ≈ 80 MB, not the 3.2 GB table), and
~500 validation mentions for c; decide Sun night, cut if not running by Tue noon.
