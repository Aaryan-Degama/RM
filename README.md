# Edition Confusion in Emerging-Entity Linking — team guide

**Team:** Aaryan Degama (IIB2024023), Geethika Gorakala (IIB2024024), Archee Jaiswal (IIB2024022). B.Tech. IT-BI, IIIT Allahabad.

Read this once from top to bottom before the evaluation. It explains the project in plain words, lists the numbers to remember, and gives an answer to every question we expect, including "you did nothing".

- The full details and every number: `PROPOSAL.md`.
- The report: `report/report.tex`.
- The code: `code/` (about 1,200 lines).

**One rule for all of us:** every number we say must be one that is in `PROPOSAL.md`. If you are not sure of a number, say "I will show you the table" and open it. Do not guess.

---

## 1. The project in one minute

- **Entity linking** means matching a phrase in a text to the right Wikipedia page. Example: in a football report, "Serie B" should link to the page "2020–21 Serie B".
- Wikipedia changes every year. A model trained in 2019 has never seen pages created in 2020. These are **new entities**. Pages that exist in every year are **continual entities**.
- Our base paper, **TempEL** (NeurIPS 2022), shows that models get much worse on new entities over time (up to 17.9 points lost, against 3.1 for continual ones).
- **What we found:** many new entities are the next *edition* of something that already exists (a new season, a new election, a new tournament). The model often does find the right page, but puts a *different edition* at rank 1. We call this **edition confusion**.
- **What we built:** a small rule that re-orders the model's list. Inside a group of sibling pages (a "family"), it puts first the edition whose year appears in the text. It needs **no training**.
- **What we also found:** two later papers (TIGER, CYCLE) report a "new entity" score, but their test set is **97% old entities**. We re-ran TIGER's code and scored new entities separately for the first time.

---

## 2. The story, step by step

### 2.1 Base paper: TempEL
- 10 yearly Wikipedia snapshots, 2013–2022.
- Per year: 10,000 continual entities and 373 new entities in the test set (46,765 and 1,450 mentions).
- Model: a BLINK bi-encoder on BERT-large. It turns the text into a vector and every Wikipedia page into a vector, then picks the closest pages.
- Metric: **accuracy@64** = is the right page among the top 64 candidates?
- TempEL says about 90% of new entities are either a new edition of a recurring event (~68%) or a page split from an older page (~22%).

### 2.2 The problems in the existing work
1. TempEL **guesses** that similar editions cause the errors but never measures it.
2. TIGER and CYCLE report "new entity" results on a test set of 46,765 continual + 1,450 new mentions. So the score mostly measures old entities. We checked this in their code and data.
3. TIGER and CYCLE search among only **10,373** pages. TempEL searches among **6,180,587**. A small pool is much easier.
4. CYCLE's public code cannot run (a folder is missing), so nobody can re-check it.

### 2.3 What we focus on
- The **ranking** step: the right page is in the list but not at rank 1.
- **New entities that belong to a family** (editions). Not COVID-type entities, which are truly new and need new knowledge.
- **No training**, because we have one laptop without a GPU, and because a rule that works without training is easy to check and to reuse.

### 2.4 Our method (five steps)
1. **Slots.** In a page title, a year, a season ("2019–20"), an ordinal ("23rd") or a number is a *slot*. "2022 FIFA World Cup" → template "# FIFA World Cup".
2. **Families.** Candidates in the list with the same template form a family. The page without the slot ("FIFA World Cup") joins it.
3. **Evidence.** For each family member, check if its slot value (for example "2020–21") appears in the text around the mention.
4. **Score.** `z = s − (λ/2)·‖g‖² + η·c·w`
   - `s` = the model's own score, `g` = the page's vector, `w` = the evidence (0, ½ or 1).
   - Chosen settings: evidence first (η = ∞), then the corrected score (λ = 0.5).
5. **Reorder inside the family only.** A family keeps the same list positions; only the order inside it changes. Pages outside a family never move.

**Why this is safe:** it cannot push the right page out of the list, and it cannot disturb pages that are not siblings. The code checks these guarantees on every run.

**The example to show** (test mention 2538):

| Rank | Before (TempEL model) | After our reorder |
|---|---|---|
| 1 | Serie B | **2020–21 Serie B** ✔ |
| 2 | Serie B (disambiguation) | Serie B (disambiguation) |
| 3 | 2018–19 Serie B | 2019–20 Serie B |
| 4 | Lega B | Lega B |

The right page was at rank 17. The text says "2020–21", so it moves to rank 1. Ranks 2 and 4 are not in the family and do not move.

### 2.5 How we tested it honestly
- We wrote the rules of the test and **committed them to git before looking at the test data** (commit `12fd858`, 7 Nov 2026). This is called **pre-registration**. It stops us from picking whichever result looks best.
- The test data was scored **once**.
- We compared against **controls**: simple rules that could explain the gain without our idea.

| Control | The question it answers |
|---|---|
| Recency prior (pick the newest edition) | Is our gain just "pick the latest"? |
| Snapshot-year prior (pick the 2020 edition) | Is our gain just the benchmark's year? |
| Placebo (use the text of a different mention) | Does the mention's own text matter? |
| Title-matcher (year match on the whole list, no families) | Do families matter? |

---

## 3. The numbers to remember

### 3.1 On TempEL (full pool of 6.18 million pages; 2019 model tested on 2021) — our main result

| What | Number |
|---|---|
| Our reproduction of TempEL, accuracy@64, new / continual | **0.7345 / 0.7673** (paper: 0.734 / 0.767) |
| Recurrent new entities: wrong edition at rank 1 | **18.7%** (right page at rank 1: only 12.8%) |
| New-entity accuracy@1, before → after our method | **0.174 → 0.210** (+3.6 points, 95% CI [+1.8, +5.6], p = 0.0001) |
| Old entities our method can reorder | **+2.5 points** |
| New-entity mentions fixed / broken by our method | **52 / 0** |
| Best possible within-family reorder (ceiling) | 0.257 |
| Accuracy@64 after re-ordering the top 256, new entities | **0.751** (TempEL publishes 0.734) |

How the controls did on new-entity accuracy@1: recency +0.7, placebo −0.3, title-matcher +1.0, snapshot-year **+3.2**, ours **+3.6**.

### 3.2 On TIGER's setup (small pool of 10,373 pages; R@1; our re-run of their public code, one seed)

Models trained in 2019, tested in 2022:

| Model | Published (pooled) | Our re-run, pooled | **New entities only** | Continual only |
|---|---|---|---|---|
| BLINK | .142 | .193 | **.343** | .189 |
| TIGER | .217 | .199 | **.349** | .195 |
| CYCLE | .222 | cannot run | – | – |

- The pooled score ≈ the continual score (they differ by at most 0.6 points).
- TIGER minus BLINK in our re-run: +0.5 to +0.7 points pooled; on new entities alone it cannot be told apart from zero.
- Our method on top of these models: +0.4 to +2.2 points on new entities, about +2 on continual ones.
- **A plain title-matcher does better than our method here: +5 to +12 points.**

---

## 4. The two weak points — say them first, before the panel does

1. **The "pick the 2020 edition" rule ties us on new entities** (+3.2 vs our +3.6; the difference is not significant).
   - Why: TempEL's new entities are pages created in the last year, and 81% of the dated ones are 2020 editions. So "pick 2020" is almost a cheat code for this benchmark.
   - Our answer: that rule **hurts old entities** (−0.3) because it ignores the text. Ours **helps them** (+2.5) because it reads the year from the text. Ours is **the only tested rule that helps both groups**.
2. **On TIGER's small pool, a plain title-matcher beats our method** (+5 to +12 vs +0.4 to +2.2).
   - Our reading (not tested): among only 10,373 pages, a page whose year matches the text is almost always right. Among 6.18 million pages there are many wrong pages with a matching year, and there the title-matcher gains only +1.0 while ours gains +3.6.
   - This is also a finding: on the small pool, a rule with no training gains more than the difference between TIGER and BLINK. The small pool is very easy to move with simple title tricks.

Saying these ourselves is a strength. It shows we tested our own idea properly.

---

## 5. What we must NOT say

- ❌ "Our method is state of the art."
- ❌ "We beat TIGER / CYCLE." (Ours is a re-ordering step on top of a model, not a competing model.)
- ❌ "TIGER's paper is wrong." ✔ Say: "The published gap did not reproduce in our one-seed re-run of their public code."
- ❌ "Our method beats every baseline." ✔ Say: "It is the only tested rule that helps both new and old entities."
- ❌ Anything about fixing COVID entities. Our method does not help them, and we say so.
- ❌ Any number that is not in `PROPOSAL.md`.

---

## 6. "You did nothing" — the answer

**Short answer:** "We did not train a new big model, and that is on purpose. Here is what we did do." Then list:

1. **Reproduced the base paper on the full 6.18-million-page pool, on a laptop.** Downloaded about 28 GB (the 2019 model and the 2021 page vectors), encoded all 48,215 test mentions (about 3 h 10 min on CPU), and ran an exact search over 6.18M × 1024 numbers (about 55 min). Our result matches the paper to three decimals (0.7345 / 0.7673 vs 0.734 / 0.767). TIGER and CYCLE did not use this full pool.
2. **Found an evaluation problem by reading TIGER's code and data:** the "new entity" test set is 97% old entities. No paper reports a score on new entities alone for these models. We produced it.
3. **Measured something the base paper only guessed:** 18.7% edition confusion against 12.8% correct for recurrent new entities.
4. **Designed a method and changed it when it was wrong.** Our first design (removing a shared direction) was shown in review to remove the very differences between siblings. We dropped it and built the simpler reorder. The record is in `docs/ideas/review-round1.md`.
5. **Tested it like a real experiment:** pre-registered in git before the test, scored once, four controls and two ablations, confidence intervals over entities, and we report the results that go against us.
6. **Trained 16 models with TIGER's code on Kaggle** (BLINK and TIGER × new/continual × 4 years), about 2.6 hours per run. The first run crashed on the 16th model because of Kaggle's 20 GB disk limit; we fixed the script and ran it again. The two runs agree to four decimals on 43 of 46 shared scores.
7. **Wrote about 1,200 lines of code** (`code/`): encoding, exact retrieval, family building, the reorder, the statistics, the Kaggle script, and a live demo.

**If they say "you only re-ordered a list":** "Yes, and that is the result. A rule with no training gives +3.6 points on new entities and +2.5 on old ones, on the realistic full pool. We also show exactly how far it can go: the ceiling is 0.257."

**If they ask "did you use AI tools?":** answer truthfully. An AI coding assistant helped write the code and draft the documents. The design choices were ours (lead on the full TempEL pool, pre-register new-entity accuracy@1, run the full 16-model TIGER grid, keep one method), we ran the jobs, and every number comes from a result file we can open in front of the panel.

---

## 7. Likely questions and answers

**Q1. What is the difference between a new and a continual entity?**
A continual entity has a Wikipedia page in every snapshot from 2013 to 2022. A new entity's page was created in the year before the snapshot. In the 2021 test set there are 373 new entities with 1,450 mentions.

**Q2. Why is accuracy@1 so low (0.17–0.21)?**
The model must pick one page out of 6.18 million, and it was trained on 2019 data but tested on 2021. TempEL itself only reports accuracy@64 (0.734 / 0.767), which is much higher because the right page only has to be in the top 64.

**Q3. Why did you not train a model?**
Three reasons. (1) One laptop, no GPU. (2) The panel ruled out building a dataset and our mentor told us to reuse TempEL, as TIGER and CYCLE did. (3) A training-free rule isolates the cause: if a simple reorder fixes errors, the errors really are edition confusion.

**Q4. What exactly is new in your work?**
(1) The first new-entity-only scores for BLINK and TIGER under TIGER's protocol. (2) The first measurement of where new entities fail in the ranking. (3) The within-family reorder, tested with pre-registration and controls.

**Q5. +3.6 points is small. Why does it matter?**
It is +3.6 points on a base of 17.4, with no training and no loss on old entities. The best possible within-family reorder reaches 0.257, so we recover a bit under half of the available room (0.174 → 0.210 of a possible 0.257). We do not claim more than that.

**Q6. A simple "pick the 2020 edition" rule does the same. So what is your contribution?**
See section 4, point 1. That rule works only because of how the benchmark was built, and it hurts old entities. Ours uses the text and helps both.

**Q7. Why does the title-matcher beat you on TIGER's pool?**
See section 4, point 2.

**Q8. Did you prove TIGER wrong?**
No. In our re-run TIGER is still ahead of BLINK, by 0.5 to 0.7 points instead of the published 5 to 7.5. We ran one seed of the re-released code on different hardware, and the bigger mismatch is in the BLINK baseline row (.193 in our run, .142 published). We report it and do not explain it.

**Q9. Why is the score on new entities higher (0.34) than the pooled score (0.19) in TIGER's setup?**
Because the pooled score is 97% continual mentions, and in that small pool the new-entity mentions happen to be easier than the continual ones. This is exactly why they must be scored separately.

**Q10. Why only one seed?**
Kaggle gives 30 GPU hours a week and one full grid takes about 2.6 hours. We chose to cover all 16 models once. We state "one seed" everywhere.

**Q11. Did you tune on the test set?**
No. The two settings (η, λ) were chosen on the validation set, in five folds, and the rules were committed to git before the test set was scored. The test set was scored once.

**Q12. What is a confidence interval "over entities"?**
There are only 373 new entities but 1,450 mentions, and mentions of the same entity are not independent. So we resample whole entities, which gives wider and more honest intervals.

**Q13. What is the p-value of 0.0001?**
If the method did nothing, a gain as large as +3.6 would appear in about 1 of 10,000 random sign flips. It is from a permutation test with 10,000 permutations.

**Q14. What about COVID entities?**
Not helped, and we say so. They have no editions (0.0% edition confusion), and for 37.7% of their mentions the right page is not retrieved at all. They need a method that adds knowledge, such as RAED.

**Q15. Does it break old entities?**
On old entities it fixed 424 mentions and broke 168, a net gain. On new entities it fixed 52 and broke 0.

**Q16. Why not use an LLM?**
An LLM trained after 2021 has already seen the "future" pages, so its result would be contaminated. Everything we use is dated on or before 1 Jan 2021.

**Q17. Why did you not compare with RAED?**
RAED uses a search index from January 2023, which is later than the test years, and it reports accuracy@64 only. The numbers are not comparable, so we do not put them side by side.

**Q18. Can you show the retrieval?**
Yes. See section 8.

**Q19. What would you do next?**
Other years on the full pool; fixing the cases where the right page is missing but a sibling is in the list (10.5% of recurrent new mentions); sending COVID-type entities to a knowledge-adding method; training with sibling pages as hard negatives.

**Q20. What is still unfinished?**
A manual check of 100 sampled errors (to confirm the automatic error labels). No number in the report depends on it.

---

## 8. Live demo

From the repository root:

```bash
PYTHONPATH=code .venv/bin/python code/demo.py 2538        # the Serie B example
PYTHONPATH=code .venv/bin/python code/demo.py "world cup" # search by word
PYTHONPATH=code .venv/bin/python code/demo.py             # interactive; Enter = a random corrected example
```

- It shows the model's top 10 next to the list after our reorder, for any of the 48,215 test mentions.
- It **replays saved results**. It cannot take a new sentence typed by the professor. If asked, say so: live search over 6.18 million pages was not built; it needs the 25 GB table of page vectors and the model loaded.
- It needs the `data/` folder, which is on Aaryan's laptop only (it is too big for git).

---

## 9. Words you should be able to explain

| Word | Meaning |
|---|---|
| Mention | The phrase in the text that we want to link ("Serie B") |
| Entity | A Wikipedia page |
| Pool | All the pages the model can choose from (6,180,587 in TempEL 2021; 10,373 in TIGER's setup) |
| Bi-encoder | A model with two encoders: one for the text, one for the pages; closeness = dot product |
| accuracy@1 / R@1 | The right page is at rank 1 |
| accuracy@64 | The right page is in the top 64 |
| MRR | Average of 1/rank of the right page |
| Frozen | The model is used as released; we do not train it |
| Family | Pages in the list that share a title template ("# Serie B") |
| Umbrella page | The family's page without a year ("Serie B") |
| Edition confusion | Rank 1 is a sibling of the right page, not the right page |
| Control | A simpler rule tested to rule out an easy explanation |
| Pre-registration | Writing down the test rules before seeing the test data |
| Pooled | New and continual mentions scored together |
| Gap | Test year minus training year |

---

## 10. Timeline (for "when did you do this?")

| Date (2026) | What |
|---|---|
| 27 Sep | First proposal |
| 6 Nov | Final method design after review; TIGER/CYCLE code and data checked |
| 7 Nov | TempEL reproduced on the full pool; test rules committed (`12fd858`); the single test pass |
| 11 Nov | Depth-256 table (accuracy@64) |
| 15 Nov | TIGER grid on Kaggle (two runs); new-only scores; our method on TIGER's lists |

---

## 11. Who did what

Fill this in together, truthfully, before the evaluation. Each person should be able to explain their own rows without notes.

| Part | Who |
|---|---|
| Reading the papers (TempEL, TIGER, CYCLE, RAED) | |
| TempEL reproduction (download, encoding, retrieval) | |
| Error diagnosis and family rules | |
| Method, controls and the pre-registered test | |
| TIGER re-run on Kaggle and re-scoring | |
| Hand audit of 100 errors | |
| Report and slides | |
