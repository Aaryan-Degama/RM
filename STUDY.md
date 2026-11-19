# Study notes — everything I need to know for the evaluation

Personal notes for Aaryan. `README.md` is the short team guide with 20 questions and answers; this file goes deeper. Every number here is from `PROPOSAL.md`.

---

## 0. What changed since the last evaluation (read this first)

Last time I told the panel about a **prototype + residual** idea and about combining our method with TIGER and RAED. This is what happened to each part.

| What I said last time | Now | Why |
|---|---|---|
| Find the **family** of a mention | **Kept.** It is the core of the method. | |
| Find the family with a **prototype vector** and dot products | **Changed.** Families come from page **titles** ("# Serie B"). | Titles are exact and need no tuning; a vector prototype for 2–10 siblings is too noisy. |
| Use the **residual**: strip the common part of siblings, compare what is left | **Dropped.** | The review (6 Nov) showed the math removed the directions in which siblings **differ**, which is the opposite of the goal. For a family of two it is exactly a penalty on vector length (checked to 1e-13). |
| What is left of the residual | A small **length-correction term** (λ) in the score | On its own it gains nothing (−0.1 points). With the year evidence it adds about 0.6. |
| Decide "sibling or continual" from how similar the pages are | **Dropped.** | Not needed: we only reorder inside a family, we don't classify. |
| Route COVID-type entities to **RAED** | **Future work.** | But we measured the reason for it: COVID entities have 0.0% edition confusion and 37.7% are not retrieved at all. |
| Use a **graph-based** technique | **Not used.** | Our method uses no graph. TIGER does; we re-ran TIGER. |
| **Train** the reranker (the mentor's suggestion) | **Planned, not done: not enough time.** | It needs page text, a separate training split and GPU time: about 2–4 weeks part-time (my estimate). We first measured how much room there is (the oracle, below). |

**If the panel asks "where is the residual?"**
"We tested that idea on paper before running anything. The residual removed the very differences between editions, so it could not separate them. We replaced it with direct evidence: does the year in the page title appear in the text? That is what gives the gain. We kept the record of this decision in `docs/ideas/review-round1.md`."

**Two things I said last time that were wrong. Do not repeat them.**
- ✗ "90% of new entities are actually continual entities." ✓ About 90% of new entities are **new pages in an existing family** (next edition ~68%, page split ~22%). They are still new entities.
- ✗ "TIGER enriches the mention." ✓ TIGER uses graphs during **training** to shape the **entity** vectors. It does not add information to the mention.

---

## 1. The project in five sentences

1. Models that link text to Wikipedia pages get worse over time, mostly on **new** pages (TempEL: up to 17.9 points lost for new entities, 3.1 for old ones).
2. Two later papers (TIGER, CYCLE) report a "new entity" score, but their test set is **97% old entities**; we re-ran TIGER's code and scored new entities alone (0.34 against the pooled 0.19).
3. On TempEL's full pool we measured **why** new entities fail: for recurring events, a **wrong edition** is at rank 1 for 18.7% of mentions and the right page for only 12.8%.
4. We built a **training-free reorder** inside title families; in a pre-registered test it lifts new-entity accuracy@1 from 0.174 to 0.210 and also helps old entities (+2.5).
5. We report what goes against us: "pick the 2020 edition" ties us on new entities, and on TIGER's small pool a plain title match beats us.

---

## 2. Background I must be able to explain

- **Entity linking:** match a phrase in a text (a *mention*) to the right Wikipedia page (an *entity*).
- **Bi-encoder (BLINK):** two BERT encoders. One turns the mention + 128 tokens of context into a vector `f`. The other turns each page's description into a vector `g`. Score = dot product `s = f · g`. The pages with the highest scores are the candidates.
- **Pool:** all pages the model can choose from. TempEL 2021: **6,180,587**. TIGER's setup: **10,373**.
- **accuracy@1 / R@1:** the right page is at rank 1. **accuracy@64:** the right page is in the top 64. **MRR:** average of 1/rank.
- **Frozen:** we use the model exactly as released. We train nothing on the TempEL side.
- **Continual entity:** has a page in every snapshot 2013–2022. **New entity:** its page was created in the year before the snapshot.
- **Snapshot:** Wikipedia as it was on 1 January of a year. Our test: model trained on the **2019** snapshot, tested on the **2021** snapshot.

---

## 3. The base paper: TempEL (NeurIPS 2022)

- 10 yearly snapshots. Per snapshot in the test set: 10,000 continual entities (46,765 mentions), 373 new entities (1,450 mentions).
- Published metric: accuracy@64. For the 2019 model on 2021: **0.734 new, 0.767 continual**.
- Findings we build on:
  - New entities lose up to 17.9 points over time, continual up to 3.1.
  - ~68% of new entities are recurring events, ~22% are page splits.
  - COVID: 24% of 2021 new entities, 30% of their mentions, ~14 extra points lost.
- What it does **not** do: it guesses that similar editions cause the errors (appendix) but never measures it. It reports no accuracy@1.

**Easy to get wrong:** new entities *degrade* more, but their absolute accuracy@64 is not always lower than continual.

---

## 4. TIGER and CYCLE, and what we found in them

- TIGER (ECAI 2024) and CYCLE (CIKM 2024) train the bi-encoder with graph information. They report R@1 on a reduced TempEL.
- Published R@1, trained on new-entity mentions, tested 3 years later: BLINK **.142**, TIGER **.217**, CYCLE **.222**.
- Four facts we verified in their code and data:
  1. The "new entity" columns are from models **trained** on 1,764 new mentions but **tested** on 46,765 continual + 1,450 new mentions. 97% continual.
  2. The pool is 10,373 pages, and every page in it is the answer to some test mention. Few distractors.
  3. "Gap 3" is one single pair: train 2019, test 2022.
  4. CYCLE's public code cannot run (a source folder is missing). We quote it; we cannot re-run it.

---

## 5. The method, with the reasoning

### 5.1 Slots and templates
- A **slot** is a title word that is a year, a season ("2019–20"), an ordinal ("23rd"), a number, a Roman numeral (2+ letters), or a number word.
- Mixed words like "COVID-19" or "F-16" are **never** slots.
- Replace slots by `#` → the **template**. "2022 FIFA World Cup" → "# FIFA World Cup".

### 5.2 Families
- Candidates **in the top-64 list** with the same template form a family.
- The **umbrella page** (the template with no slot: "FIFA World Cup") joins.
- A family needs at least 2 members, one with a slot.

### 5.3 Evidence `w`
For a family member with a slot, look in the 128 tokens the encoder saw:
- slot value inside the mention itself → `w = 1`
- slot value elsewhere in the window → `w = ½`
- a bare year that is only one end of a season → counts half
- non-year slots count only inside the mention
- umbrella page → `w = 0`

### 5.4 The score

`z(e) = s(e) − (λ/2)·‖g‖² + η·c·w(e)`

- `s` = the model's score. `g` = the page vector. `w` = the evidence.
- **Why the middle term.** Distance between mention `f` and page `g`:
  `‖f − g‖² = ‖f‖² − 2·(f·g) + ‖g‖²`.
  `‖f‖²` is the same for every candidate, so "smallest distance" = "largest `s − ½‖g‖²`". So λ = 1 ranks by distance, λ = 0 by dot product. The dot product favours pages with long vectors; the term removes that advantage.
- **`c`** = the typical gap between the 1st and 2nd score on validation (2.10). It only sets the unit of η. It needs no labels.
- **Chosen values** (on validation, the same in all 5 folds): **η = ∞, λ = 0.5**. In words: inside a family, members whose year is in the text go first; the corrected score breaks ties.

### 5.5 The ordering
Each family keeps **the list positions it already holds** and only the order inside changes. Everything outside a family stays where it is.

### 5.6 The four guarantees, and why each is true
1. **Non-family candidates keep their rank.** We never touch their positions.
2. **accuracy@64 cannot change at depth 64.** We only shuffle the same 64 pages; whether the right page is among them is unchanged.
3. **Any change at rank 1 is a swap between siblings.** Only family members move, and only into family positions.
4. **λ = η = 0 gives the original list.** Then `z = s`, and the list was already sorted by `s`.

### 5.7 Which term does the work?
| Version | New acc@1 gain |
|---|---|
| λ only (length correction) | −0.1 |
| η only (year evidence) | +3.0 |
| Both (ours) | +3.6 |

The gain comes from the year evidence. The length term adds about 0.6 on top (not tested for significance).

---

## 6. The controls, and what each one proves

| Control | What it does | Result on new acc@1 | What that proves |
|---|---|---|---|
| Recency prior | Inside a family, pick the newest edition | +0.7 | "Pick the latest" is not enough |
| Snapshot-year prior | Inside a family, pick the 2020 edition | **+3.2** | The benchmark's year explains most of the gain on new entities (see §9) |
| Placebo | Use the text of a **different** mention as evidence | −0.3 | The gain needs **this mention's own text** |
| Title-matcher | Year match on the **whole list**, no families | +1.0 | **Families matter**: without them the gain is a third |

---

## 7. How the test was run, and why

- **Pre-registration.** The rules (data, metric, test, controls) were committed to git as `12fd858` on 7 Nov, **before** the test set was scored. Reason: so nobody can say we picked the best-looking result afterwards.
- **One test pass.** The test set was scored once. Never rerun it.
- **Cross-fitting in 5 folds.** Validation and test share **every** entity (373/373 new). If we tuned on all of validation, the settings could fit those exact entities. So entities are split into 5 groups by title template, and a mention is scored with settings chosen on the other 4 groups.
- **Primary test.** Change in new-entity accuracy@1. **Sign-flip permutation test**, 10,000 permutations: if the method did nothing, each entity's change would be equally likely to be + or −; we flip signs at random and see how often the gain is as big as ours. Result: p = 0.0001.
- **Intervals over entities.** There are 1,450 new mentions but only 373 entities, and mentions of the same entity are not independent. So the bootstrap resamples **entities**. This gives wider, more honest intervals.
- **Non-inferiority.** For old entities: the method passes if the lower end of the interval is not below −1.0 point. It was **+1.9**.
- **Holm correction.** We ran 12 secondary comparisons. With many tests, some look significant by luck; Holm raises the bar for each.
- **Temporal rule.** Everything used at test time is dated on or before 1 Jan 2021. No LLM (an LLM trained later has already seen the "future" pages).

---

## 8. Results on TempEL (full pool, 2019 model → 2021 test)

### 8.1 Reproduction
| accuracy@64 | Ours | Paper |
|---|---|---|
| New | 0.7345 | 0.734 |
| Continual | 0.7673 | 0.767 |

Frozen accuracy@1 (the paper does not report it): **0.1738 new, 0.2117 continual**.

### 8.2 Why new entities fail
| Slice | n | Correct @1 | Edition confusion @1 | Other error (gold in top 64) | Not retrieved |
|---|---|---|---|---|---|
| New: recurrent | 561 | 12.8% | **18.7%** | 48.0% | 20.5% |
| New: COVID | 440 | 17.0% | 0.0% | 45.2% | **37.7%** |
| New: other | 449 | 23.4% | 3.6% | 49.9% | 23.1% |
| New: all | 1,450 | 17.4% | 8.3% | 47.7% | 26.6% |
| Continual | 46,765 | 21.2% | 2.8% | 52.8% | 23.3% |

("Not retrieved" adds the two miss columns of the report.)

- Recurrent: a wrong edition wins more often than the right page.
- COVID: no edition confusion; the problem is retrieval. A reorder cannot help.
- **Ceiling:** a perfect family reorder would reach **0.257**. About half of all errors are other ranking errors that we cannot reach.

### 8.3 The main test
| Row | New acc@1 | Gain | Fixed / broken (new) | Gain on old entities it can reorder |
|---|---|---|---|---|
| Frozen TempEL | 0.1738 | – | – | – |
| Recency prior | 0.1807 | +0.7 | 13 / 3 | −0.2 |
| Snapshot-year prior | 0.2062 | +3.2 | 47 / 0 | −0.3 |
| Placebo | 0.1710 | −0.3 | 2 / 6 | −0.3 |
| Title-matcher | 0.1834 | +1.0 | 23 / 9 | +2.4 |
| **Ours** | **0.2097** | **+3.6 [+1.8, +5.6]** | **52 / 0** | **+2.5 [+1.9, +3.2]** |

- New edition confusion drops from 8.3% to 4.8%.
- Old entities overall: 424 fixed, 168 broken. Net gain, but **not** "no errors".
- Against controls (Holm-corrected): beats recency (+2.9, p = 0.001), placebo (+3.9, p = 0.001), title-matcher (+2.6, p = 0.02). **Does not beat snapshot-year** (+0.3, not significant).

### 8.4 On the paper's own metric (reorder the top 256)
| accuracy@64 | New | Continual |
|---|---|---|
| Paper | 0.734 | 0.767 |
| **Ours** | **0.751** (+1.7, p = 0.002) | 0.768 (unchanged) |
| Snapshot-year prior | 0.764 | 0.767 (slightly worse) |

Why this can change when §5.6 says it cannot: here we reorder the top **256**, so a family member at rank 100 can move into the top 64.

---

## 9. The two weak points, with the full explanation

**1. "Pick the 2020 edition" ties us on new entities (+3.2 vs +3.6) and beats us at accuracy@64 (0.764 vs 0.751).**
- Cause: TempEL's new entities are pages created in the last year. We measured that **81%** of the dated new correct titles are 2020 editions. So "pick 2020" is nearly always right *for new entities in this benchmark*.
- Where it fails: old entities. A text about the 2014 edition should link to the 2014 page; "pick 2020" gets it wrong (−0.3). Ours reads "2014" from the text (+2.5).
- The claim: ours is **the only tested rule that helps both groups**.

**2. On TIGER's pool, the plain title-matcher beats us (+5 to +12 vs +0.4 to +2.2).**
- On TempEL's full pool it is the reverse (+1.0 vs +3.6).
- Our reading (**not tested**): in 10,373 pages that are all answers, a page whose year matches the text is almost always right. In 6.18 million pages, many wrong pages also carry that year, so matching on the whole list makes mistakes (23 fixed, 9 broken) and families protect against that.

---

## 10. The TIGER re-run (their code, their data, one seed, Kaggle T4)

- 16 models: {BLINK, TIGER} × {trained on new, trained on continual} × 2019–2022. About 2.6 h per run.
- The first run crashed on the 16th model (Kaggle's 20 GB output limit). Fixed and re-run. Both runs agree to 4 decimals on 43 of 46 shared values.

| 2019 model → 2022 test | Published | Our pooled | **New only** | Continual only |
|---|---|---|---|---|
| BLINK | .142 | .193 | **.343** | .189 |
| TIGER | .217 | .199 | **.349** | .195 |

- **Pooled ≈ continual** (differ by ≤ 0.6 points in any pair).
- **TIGER − BLINK:** +0.5 to +0.7 pooled (significant, small). On new only: −1.9 to +0.7, not different from zero.
- **Never say TIGER is wrong.** Say: "the published gap did not reproduce in our one-seed re-run; the bigger mismatch is in the BLINK row."
- **Our method on their lists** (settings unchanged): new R@1 up in all 8 model-year cells (+0.4 to +2.2), continual up ~2 points.

---

## 11. Hard questions I should answer without notes

**Why can re-ordering the top 64 never change accuracy@64?**
It only shuffles the same 64 pages. The right page is in or out of that set before we start.

**What does the placebo prove?**
With another mention's text as evidence the gain disappears (−0.3). So the gain comes from the mention's own context, not from the act of shuffling families.

**Why does "pick 2020" tie us on new but lose on old?**
New entities in TempEL are mostly 2020 editions (81%), so the rule is right by construction. Old mentions refer to older editions, and the rule ignores the text.

**Why resample entities and not mentions?**
Mentions of one entity succeed or fail together. Treating them as independent makes the interval too narrow.

**Why 5 folds instead of tuning on validation directly?**
Validation and test contain the same entities. Folds make sure a mention is never scored with settings tuned on its own entity.

**Why is the gain on test (+3.6) smaller than on validation (+6.4)?**
That is what honest testing looks like: validation is where settings were chosen. We report both.

**Why not a cross-encoder / why didn't you train the reranker?**
We planned it, but there was not enough time this semester. It needs three things we did not have ready: the text of each candidate page (we only have titles and vectors), top-64 lists for a separate training split, and GPU time (Kaggle allows 30 h a week, 12 h per session). That is about 2–4 weeks part-time (my estimate). So we did the training-free version first, which also measures how much a reranker can gain:
- original model 17.4% → our rule 21.0% → perfect within-family reorder (oracle) **25.7%** new-entity acc@1 (`data/enc/test2021full.results_d64.json`).
- Any reranker that only fixes editions, trained or not, is capped at 25.7%. Our rule gets 3.6 of the 8.3 possible points.
- A trained reranker over all 64 candidates could go beyond 25.7%, because it can also fix errors outside families. No prediction of how much.
Full plan: §16.
Say: "A trained reranker is the next step, and it would likely do better. We stayed training-free this semester for time and compute, and because a rule with no training cannot overfit the test set."

**What would make the method fail?**
No year in the text; titles with no shared template (most page splits); truly new entities like COVID.

**Is the residual still there?**
No. See §0.

---

## 12. Never say

- "our model" (say "our method" or "our reorder")
- "state of the art", "better than TIGER/CYCLE"
- "TIGER's paper is wrong"
- "beats every baseline"
- "fixes COVID entities"
- "90% of new entities are continual"
- any number not in `PROPOSAL.md`

---

## 13. What I can show live

```bash
PYTHONPATH=code .venv/bin/python code/demo.py 2538   # Serie B: rank 17 -> 1
PYTHONPATH=code .venv/bin/python code/demo.py        # Enter = random corrected example (52 exist)
git show 12fd858 --stat                              # the pre-registration, dated 7 Nov
```

- The demo replays saved results for the 48,215 test mentions. It cannot take a new typed sentence.
- Result files: `data/enc/test2021full.results_d64.json` (main test), `data/kaggle/c1_results.json` and `tiger_b.json` (TIGER).
- Code to open if asked: `code/families.py` (families), `code/rerank.py` (reorder and statistics).

---

## 14. Still open

- Hand audit of 100 errors (`data/audit/audit_sheet.csv`).
- The slides.
- Compile the report on Overleaf.
- Push to GitHub, and decide what the repo's front page shows.

---

## 15. Cross-question bank (instant answers)

Short answers, grouped by topic. If a question is not here, the honest fallback is: **"We did not measure that; it is outside what we tested."** That is a complete answer. Never invent a number.

### A. Basics

- **What is entity linking used for?** Search engines, question answering, building knowledge graphs: anything that must know *which* "Serie B" or *which* "Paris" a text means.
- **How is it different from NER?** NER finds the phrase and its type (person, place). Entity linking decides which exact page it is. In TempEL the mention is already marked.
- **What is an embedding?** A list of numbers (here 1,024) that represents a text, so that similar texts get similar lists.
- **What is BERT?** A transformer language model pretrained on text. BERT-large has 24 layers and a 1,024-dimension output. Ours was pretrained on 2018 Wikipedia.
- **Why two encoders and not one?** So all 6.18M page vectors can be computed once and stored. Only the mention is encoded at query time.
- **Why dot product, not cosine?** The model was trained with dot product. Changing it would mean a different model. Our λ term is a controlled step towards distance, used only inside families.
- **Why top 64?** It is the benchmark's convention (BLINK and TempEL): the first stage hands 64 candidates to a second stage.
- **Why 128 tokens of context?** TempEL's own encoder setting. We use the same so the reproduction matches.
- **What is "exact search"?** We compute the score against every one of the 6.18M pages. No approximate index (like FAISS), so no approximation error.
- **How did 25 GB fit in 15 GB RAM?** The table is memory-mapped and read in chunks from disk.
- **Is it a closed world?** Yes. Every test mention has its right page in the pool, as in TempEL.

### B. Data

- **Where is the data from?** The TempEL authors' public release: mentions, the 2019 model, and the 2021 page vectors.
- **Why the 2019 model on the 2021 test?** One cell was affordable (each year's table is 25 GB), it has a two-year gap, and 2019 is also the training year used for TIGER's headline result. It was fixed before the test.
- **Why only one cell?** Disk and time on one laptop. Other years are in future work. We say this in limitations.
- **How many mentions per new entity?** 1,450 mentions for 373 entities; median 2, at most 10.
- **Is the test balanced?** No: 373 new entities against 10,000 continual. That is why new entities are scored separately.
- **Which language?** English Wikipedia only.
- **What are the three slices of new entities?** Recurrent (561 mentions), COVID (440), other (449). These are TempEL's own filter types.

### C. The method

- **What if there is no year in the text?** Then no member has evidence. Only the small length correction acts, and the family order barely changes.
- **What if two editions both appear in the text?** Both get evidence. The one inside the mention (w = 1) beats the one elsewhere (w = ½); a tie is broken by the corrected score.
- **What if the right page is not in the list?** Nothing can be done by reordering. That is the "not retrieved" bucket (26.6% of new mentions).
- **What if the right page is not in any family?** It never moves, and no family member can jump over it, because families only reuse their own positions.
- **What if the general page is the right answer ("Serie B" itself)?** This is the method's main risk. If a sibling's year is in the text, the umbrella page is pushed down. 46 of the 168 errors created on continual entities are of this kind.
- **How many lists does it actually change?** The rank-1 candidate lies in a family for 10,111 of 46,765 continual mentions (about 22%). The rest are untouched at rank 1.
- **Is using titles cheating?** No. Titles are part of the knowledge base at time t, and TempEL's page encoder reads them too.
- **Then why did the model not already use the year?** It does see it, but a dense vector blurs "2018–19" and "2020–21" into almost the same point. Our diagnosis shows the result: wrong edition at rank 1 for 18.7% of recurrent mentions.
- **It is "training-free" but you tuned two numbers?** Yes. Two numbers chosen from 24 combinations (6 values of η × 4 of λ) on validation. No gradient training, no GPU, and `c` needs no labels.
- **What does η = ∞ mean in the code?** η = 1,000,000: the evidence decides first.
- **Why λ = 0.5 and not 1?** Validation chose it, in all five folds.
- **What was the selection rule?** Highest new-entity accuracy@1, as long as old entities do not drop by more than 0.5 point. Ties go to the smaller η, then the smaller λ.
- **Can the regex make mistakes?** We made slots strict for that reason: a Roman numeral must be uppercase and at least two letters ("LIV", not "CIVIL"); number words are only first to twelfth; mixed words like "COVID-19" are excluded.
- **How fast is it?** Minutes on a laptop CPU for all 48,215 mentions. Per mention it is a regex over 64 titles and a sort.
- **Does it work in other languages or knowledge bases?** Not tested.
- **Why not learn the families with a model?** A learned grouping needs training data and adds errors we cannot inspect. Title templates are exact and can be read by eye.
- **Does it help page splits (the 22%)?** Mostly not. Split pages rarely share a title template, so they form no family. This is in limitations.

### D. Statistics

- **What does the 95% interval [+1.8, +5.6] mean?** If we repeated the test with other entities drawn the same way, the gain would fall in about that range 95% of the time. It does not include 0.
- **How many bootstrap resamples?** 2,000, resampling entities, fixed random seed 0.
- **Why is p exactly 0.0001?** With 10,000 permutations the smallest possible p is about 1 in 10,001. None of the random sign flips reached our gain.
- **Why two-sided?** We allowed for the method making things worse.
- **What is a "point"?** A percentage point. +3.6 points means 17.4% → 21.0%.
- **What is the relative gain?** About 21% (0.1738 → 0.2097).
- **Why not McNemar's test?** It treats mentions as independent. Mentions of the same entity are not, so its p-value would be too optimistic.
- **Why a margin of −1.0 for non-inferiority?** It was fixed in the pre-registration, before the test. The result was far inside it (+1.9).
- **Is 373 entities enough?** It is what the benchmark has. It is why the intervals are wide and why small subsets (about 100 mentions) show nothing significant. We say so.
- **Could the result be luck?** For the main test, p = 0.0001 on a rule fixed in advance. For the subsets, yes, and we do not claim them.

### E. Results

- **Why is accuracy@1 so low when accuracy@64 is 0.73?** Picking the single right page out of 6.18M is much harder than getting it into the top 64. The paper only reports the top-64 figure.
- **Why is new-entity accuracy@1 (17.4%) lower than continual (21.2%)?** The 2019 model never saw the new pages. That is the degradation TempEL describes.
- **How many mentions did you actually fix?** 52 of 1,450 new mentions, with none broken. On continual: 424 fixed, 168 broken.
- **How far from the best possible?** A perfect family reorder reaches 0.257. We reach 0.210 from 0.174, a bit under half of that room.
- **What about the other half of the errors?** About 48% of new mentions have the right page in the list but lose to a non-sibling. A family reorder cannot reach those.
- **Did MRR improve?** New-entity MRR 0.274 → 0.317.
- **Did edition confusion drop?** New entities: 8.3% → 4.8%.
- **Overall continual accuracy@1?** 0.2117 → 0.2172.
- **Validation showed the same pattern?** Yes: recurrent new entities, 22.9% wrong edition against 9.8% correct. The validation gain was +6.4; test +3.6.
- **Why reorder the top 256?** To be measurable on the paper's metric. Accuracy@256 is 0.8566 / 0.8625, so there are right pages between rank 65 and 256 that a reorder can lift.
- **Is 0.751 better than the paper?** On the same metric and data, yes: 0.751 against the published 0.734 for new entities, continual unchanged. But "pick 2020" gets 0.764 there.

### F. TIGER and CYCLE

- **What does TIGER do?** It adds graph information (Wikidata5M, plus a graph of similar descriptions) while training, to shape the entity vectors.
- **Which code and settings?** The authors' public code at a fixed commit (`649bb19`), their released data, their two config files (BLINK baseline and TIGER), seed 52313, one epoch.
- **Did you change their code?** Only a three-line patch that saves each mention's top-64 list. Training and scoring are theirs.
- **Why is your BLINK (.193) higher than the published BLINK (.142)?** We do not know and we say so. Possible reasons, none tested: the public code is a later re-release, different hardware, one seed.
- **Why is your TIGER (.199) lower than published (.217)?** Within 2 points; same caveats.
- **So is TIGER better than BLINK or not?** In our run, yes on pooled and continual mentions, by 0.5 to 0.7 points, and the intervals exclude zero. On new entities alone, the difference (−1.9 to +0.7) cannot be told from zero.
- **Why is new-only (0.34) higher than pooled (0.19)?** The pooled figure is 97% continual mentions, and in this small pool the new-entity mentions score higher than the continual ones.
- **Does training on new entities help new entities?** Not visibly. BLINK trained on *continual* mentions scores .304 / .296 / .300 / .332 on new-only, about the same as the new-trained model. (No interval computed.)
- **What do continual-trained models score pooled?** BLINK .331–.341, TIGER .329–.333.
- **What happens with titles in their pool?** 2022 pooled R@1 rises from .193 to .208 (BLINK) and .199 to .211 (TIGER); new-only from .343 to .386 and .349 to .389.
- **Why did you not run CYCLE?** Its public repository is missing the model source folder. It cannot run.
- **Did you ask the authors?** No. We report what the public code and data give.
- **Why one seed?** 30 GPU hours a week on Kaggle; a full grid is about 2.6 hours. We covered all 16 models once.
- **How do you know the run is stable?** The failed first run and the second run agree to four decimals on 43 of 46 shared values; the other three differ by at most 0.0004.
- **Your method on their lists, exact numbers?** TIGER 2022: .349 → .371. BLINK 2022: .343 → .365. Continual R@1 rises about 2 points in every cell.

### G. Novelty and related work

- **What is new here?** (1) New-entity-only scores for BLINK and TIGER under TIGER's protocol. (2) A measured breakdown of where new entities fail on the full pool. (3) The family reorder with a pre-registered test.
- **Was edition confusion known?** TempEL's appendix guessed it. Work on event linking has noted repeating events. We are not aware of a measurement of it on TempEL's full pool.
- **How is this different from a cross-encoder re-ranker?** A cross-encoder is a second trained model that reads mention and page together. Ours has no trained part.
- **How is it different from RAED?** RAED adds knowledge for truly new entities. Ours separates look-alike editions. They address different failures.
- **Why no comparison with RAED?** Its search index is from January 2023, later than the test years, and it reports accuracy@64 only.
- **Is your rule just the title-matcher?** No. The title-matcher applies the year match to the whole list and gains +1.0 on the full pool. Restricting it to families gives +3.6.
- **Could this be published?** We make no claim. It would first need more seeds and more years.

### H. Process and decisions

- **Why did you drop the dataset idea (TempEL-Dx)?** The panel judged building a dataset infeasible, and our mentor told us to reuse TempEL as TIGER and CYCLE did.
- **Why no graph, when you mentioned one last time?** The design review showed the title family does the job directly. TIGER is the graph method, and we re-ran it.
- **What went wrong along the way?** (1) The first design (residual) was wrong and was dropped. (2) A bug in the retrieval check (the right page was compared with itself by two slightly different computations) was found and fixed before the real run. (3) The first Kaggle run hit the 20 GB disk limit.
- **How long did it take?** Proposal on 27 September; design, reproduction and test from 6 to 11 November; TIGER runs on 15 November.
- **What did it cost?** Nothing: one laptop and free Kaggle GPU time.
- **Did you use AI tools?** Yes: an AI coding assistant helped write the code and draft the documents. The decisions and the runs were mine, and every number can be opened from a result file.
- **Who did what in the team?** Answer truthfully.
- **How do we know you did not tune on the test set?** The git commit `12fd858` with the rules is dated before the test results existed, and the test script refuses to overwrite its output.

### I. Limits and next steps

- **Biggest weakness?** The benchmark's recency artifact: "pick the 2020 edition" ties us on new entities.
- **Second weakness?** On TIGER's small pool a plain title match beats us.
- **What would you do with more time?** Other years on the full pool; a retrieval fix for misses that have a sibling in the list (10.5% of recurrent new mentions); route COVID-type entities to a knowledge-adding method; a cross-encoder as an upper bound; training with sibling pages as hard negatives.
- **Your mentor said a trained reranker would do better. Why not?** Agree: it likely would. We planned it but did not have the time: it needs candidate page text, a training split and GPU time, about 2–4 weeks part-time. The oracle (25.7%) shows how much an edition-only reranker can gain; a trained one over all 64 candidates could also fix other errors. It is next semester's plan, and the test set would need a new pre-registration first, because it was already used once.
- **What is unfinished?** The hand audit of 100 errors. No reported number depends on it.

---

## 16. Next step: a trained reranker (plan only, NOT done)

This is the plan if we continue next semester. Nothing here was run, and there are no numbers to quote. Times are my estimates, part-time.

**The idea in one line:** keep the frozen bi-encoder for retrieval (top-64), and train a second model that reads the mention and each candidate page **together** and re-scores the 64.

| # | Step | What exactly | Time |
|---|---|---|---|
| 0 | **Pre-register** | Write the model, data, metrics and the one test run in a git commit *before* touching test, as with `12fd858`. The 2021 test set was already used once, so this is required. | ½ day |
| 1 | **Training data** | TempEL's **2019** train and validation mentions, the same year the bi-encoder was trained on. Everything dated on or before the snapshot; no LLM. | 1 day |
| 2 | **Candidate page text** | Title + first paragraph of each page, from TempEL's entity files: the 2019 snapshot for training, the 2021 snapshot for test. Today we only have titles and vectors. | 2–4 days |
| 3 | **Candidate lists for training** | Encode the training mentions with the frozen bi-encoder and retrieve their top-64 from the 2019 pool (another 25 GB table). On Kaggle GPU, not the laptop CPU. Siblings appear in these lists naturally, so they act as **hard negatives**. | 3–5 days |
| 4 | **Model** | A **cross-encoder**: BERT-base reads `[CLS] mention in context [SEP] page title + text [SEP]` (about 256 tokens) and outputs one score. Optional variant: add our year-evidence `w` as an extra input. | 1 day |
| 5 | **Loss** | For each mention, a softmax over its 64 candidates; the loss is −log(probability of the correct page). If the correct page was not retrieved, insert it during training only. | (part of 4) |
| 6 | **Training** | Kaggle T4, mixed precision, 1–2 epochs, about 16 candidates per mention per step (always including the siblings). Choose the checkpoint on 2019 validation only. Kaggle limits: 12 h per session, 30 GPU-hours a week. | 2–4 days |
| 7 | **Test scoring** | Score the 2021 test top-64: 48,215 mentions × 64 = about 3.1 million pairs. Final score = cross-encoder score, or a mix with the bi-encoder score chosen on validation. | 1–2 days |
| 8 | **Evaluation** | Same as now: new-entity acc@1 (primary), MRR, sibling-confusion, continual non-regression. Rows: frozen, our rule, the trained reranker, reranker + our rule, the oracle (25.7%), plus the controls. Entity bootstrap, permutation test, Holm. acc@64 cannot change, because only the top 64 are reordered. | 3–5 days |

**Total: about 2–4 weeks part-time.**

**What it could add that our rule cannot:** fix errors **outside** families, such as picking the right tournament among several 1984 golf pages. That is why it might pass the 25.7% within-family ceiling. We make no prediction of by how much.

**Risks to say if asked:**
- It may learn the 2020-edition shortcut (81% of dated new gold pages are 2020 editions), so the recency and snapshot-year controls stay in.
- One seed on a free GPU, same limit as our TIGER re-run.
- Page text from a later dump would leak the future. Use only the snapshot's own text.

**One sentence for the panel:** "Next we train a cross-encoder on the 2019 data that re-scores the same top-64. It is pre-registered and judged against our rule, the controls and the 25.7% oracle. It needs about a month, mainly for page text, training lists and GPU time."

### 16.1 The quick version: a small learned reranker (also NOT done)

If a prof asks "could you not train something small right now?", this is how we would do it. It fits in a few hours and needs no GPU, because it uses only numbers we already store.

1. **Pre-register.** A git commit with the features, the model and the single test run, before anything is scored.
2. **Data.** The 2021 **validation** lists (the same data the rule was tuned on), grouped into the same 5 template folds. Never the test set while building it.
3. **Features per candidate** (all already stored): the frozen score s; its gap to the top score; the rank; "in a family" (0/1); the year evidence w (0, ½, 1); the vector length ‖g‖²; the family size; whether the page is the umbrella page.
4. **Model.** Logistic regression, or a small gradient-boosted tree model, giving one score per candidate. The list is sorted by that score.
5. **Training target.** Correct page = 1, the other 63 = 0. A softmax over the 64 (listwise) or plain 0/1 labels both work.
6. **Cross-fitting.** Train on 4 folds and score the 5th, so no mention is scored by a model that saw it.
7. **Selection rule, fixed in advance.** As for the rule: maximise new-entity validation acc@1 while continual acc@1 drops by at most 0.2 points.
8. **One test run.** Same metrics, same controls, same entity bootstrap and Holm. Report it next to the rule and the 25.7% oracle.

**Time:** about a day to build and check, and minutes to train on the laptop CPU.

**Honest limits:**
- It sees the same signals as our rule, so it mainly learns a better **mix** of them. It cannot read page text, so it cannot fix errors outside families like the 1984 golf example. Only the cross-encoder in §16 can.
- It may learn the 2020-edition shortcut. The recency and snapshot-year controls must stay in.
- Trying several versions and reporting the best on test would be tuning on the test set. That is why steps 1 and 7 are fixed before the run.

**Why we did not do it before the evaluation:** the test set had already been used once; a new result would only be exploratory; and building it would have taken the time needed for the slides and preparation. No prediction of whether it beats 21.0%.
