# Temporal Entity Linking Under Emerging Entities — a research design

**Primary benchmark:** TempEL (Zaporojets et al., NeurIPS 2022 D&B)
**Targets:** TempEL bi-encoder baseline, TIGER (Zhang, Cao, Groth 2024), CYCLE (Zhang et al., CIKM 2024)
**Related:** RAED (Ghonim et al., EMNLP 2025), EDIN, BLINKout, STAMO
**Status:** research ideation; no implementation, no performance estimates

---

## The observation this whole design starts from

Recall@1 at a three-year train–test gap, as reported by TIGER and CYCLE on their own Graph-/GCL-TempEL splits:

**Continual entities**

| Model | Recall@1 (gap=3) |
|---|---|
| BLINK | .177 |
| SpEL | .221 |
| TIGER | .304 |
| CYCLE | .301 |

**New / emerging entities**

| Model | Recall@1 (gap=3) |
|---|---|
| BLINK | .142 |
| SpEL | .192 |
| TIGER | .217 |
| CYCLE | .222 |

TIGER and CYCLE are within half a point of each other on emerging entities despite using different graph mechanisms. On continual entities they both gain roughly eight points over SpEL. Adding more graph machinery is producing diminishing returns on exactly the population we care about.

---

## 1. Problem diagnosis

The phrase "emerging entity" hides at least four different things, and they fail for different reasons. Separating them is the single most useful thing to do before designing anything.

### 1.1 What actually makes an emerging entity different

**(a) The entity-side representation is a point estimate from one noisy sample.** A continual entity's Wikipedia description has been edited for years; it has converged onto the properties that distinguish that entity from everything nearby. A newly created page is a stub written before anyone knows which properties will turn out to be distinguishing. The bi-encoder's entity vector is a deterministic function of that stub. Statistically, the entity representation has its highest variance exactly at the moment of birth — and existing methods treat it as if it were a fixed, fully-observed quantity.

**(b) New entities are born into crowds, not into empty space.** This is the part that most proposals miss. TempEL's authors manually inspected 100 randomly sampled new entities and found roughly 90% were either recurrent events (~68%, e.g. the next annual edition of a tournament) or extracts split off from already existing pages (~22%). A new entity of this kind is not an alien concept. It is a new member of a densely populated reference class, and everything in its description is shared with its siblings *except* one or two slots — a year, an edition number, a sub-team, a jurisdiction. The information that identifies it is a tiny fraction of the information that describes it.

**(c) There is no prior.** No in-link counts, no mention prior, no popularity signal. Worth noting honestly: TempEL deliberately filters out the highest-prior mentions during construction, so on this benchmark the absence of a prior hurts less than it would in the wild. Do not build the story around it.

**(d) A minority of new entities require genuinely new world knowledge.** TempEL's own analysis isolates this: COVID-19 entities make up 24% of the new entities in the 2021 snapshot and are linked by 30% of its mentions, and they carry an extra ~14 accuracy@64 point penalty over other new entities under out-of-snapshot finetuning. This is a different failure from (b) and needs a different mechanism.

> (b) and (d) are the two populations, and they are not the same problem. (b) is a *discrimination* problem inside a well-known neighbourhood. (d) is a *knowledge* problem. The graph line of work (TIGER, CYCLE) implicitly attacks representation quality in general; the retrieval line (RAED) attacks (d). Nobody has separated the populations, measured their relative mass, and routed between them.

### 1.2 Why a standard bi-encoder fails here

- **The entity encoder cannot condition on anything but the entity's own truncated description.** 128 BERT tokens, mean-reduced or CLS-reduced to a single vector. If the distinguishing token is one numeral in a stub, it is diluted by everything it shares with its siblings.
- **The training objective never sees the negatives that matter.** Both BLINK and TIGER use in-batch negatives: for a batch of N pairs the loss contrasts the gold entity against the other entities that happened to land in the batch. The probability that a random in-batch negative is the gold entity's near-duplicate sibling is essentially zero. The model is therefore never trained to place resolution along the one axis that decides emerging-entity cases.
- **Candidate density rises monotonically over time.** TempEL shows the Jaccard similarity among the top-64 returned candidates *increasing* year over year while the Wikipedia entity count grows from ~4M to ~6.4M. A fixed embedding budget has to separate more entities that are closer together.
- **The mention side drifts faster than the entity side.** TempEL's Figure 3 shows the vocabulary of mention contexts drifting substantially more than entity descriptions. Almost all of the temporal-EL literature has worked on the entity side.

### 1.3 The failure-mode decomposition

| | Failure mode | What it means operationally | Who addresses it |
|---|---|---|---|
| A | Representation | Gold is in the candidate pool but its vector is not individuating enough to win. | TIGER, CYCLE (globally, at training time) |
| B | Candidate retrieval | Gold is not in the top-K at all. Directly measurable as $1 - \text{recall@}64$. | BLINK-style dense retrieval; RAED bypasses the inventory |
| C | Temporal drift | Mention-context distribution at test time differs from training. Distinct from A. | Largely unaddressed |
| D | Knowledge availability | The concept is absent from pretraining and from the KB text. No amount of representation shaping helps. | RAED, STAMO |
| E | Ambiguity | Two or more KB entities are genuinely compatible with the context. The evidence underdetermines the answer. | Classic EL; not separated from A in temporal work |
| F | Calibration / unknown detection | No mechanism to abstain or say "not in the KB." Forced-choice architecture. | BLINKout, EDIN |

The uncomfortable fact: **no published work reports the relative mass of A–F on TempEL's emerging subset.** Everyone reports aggregate recall. Until you know whether your errors are B (retrieval) or A/E (discrimination), you cannot know whether a representation method can possibly help. Producing that decomposition is cheap, is a contribution in itself, and is the precondition for everything below.

---

## 2. What TempEL reveals

Five things, one of which is commonly misstated.

1. **Continual entities degrade slowly and monotonically; new entities degrade much harder.** Up to 3.1 accuracy@64 points for continual, up to 17.9 for new.
2. **But in TempEL's own experiments, new entities score *higher* in absolute accuracy@64 than continual entities** (e.g. 0.910 vs 0.785 for the 2013-trained, 2013-tested cell). The TempEL paper states this explicitly and explains it: most new entities are recurrent events or page extracts requiring little genuinely new knowledge. *The claim "emerging entities are harder" is true about degradation and about TIGER/CYCLE's setup; it is false about TempEL's absolute numbers.* A reviewer will catch this. State it correctly and it becomes an asset — it is precisely the evidence that most new entities are near-duplicate discrimination cases, not knowledge cases.
3. **The 2021 snapshot is the anomaly** (new-entity accuracy@64 drops to 0.688–0.761), driven by COVID entities. This is the clean empirical handle on population (d).
4. **In-snapshot finetuning helps new entities more than continual ones**, and the effect is concentrated on the COVID subset — direct evidence that the two populations respond to different interventions.
5. **The benchmark is hard by construction.** Mentions where the gold entity has the highest prior are excluded, and mentions with low normalised edit distance to the entity title are excluded. TempEL is specifically a non-trivial-disambiguation benchmark. That is good for us: it means aggregate numbers are low and the remaining errors are the interesting ones.

Two structural constraints to plan around: each snapshot's new-entity training set has only **1,764 mentions over 373 entities**, and test/validation cap mentions at 10 per entity. Anything claiming to "accumulate evidence over time" has at most ten observations per entity, all inside one yearly bucket, with no intra-year timestamps.

---

## 3. What TIGER solves

TIGER's contribution is best read not as "adding graph features" but as **shaping the geometry of the entity embedding space using a structural signal that is discarded at inference.** The paper is explicit that inference uses text only: entity description, mention, and context. The structure graph (Wikidata5M relations), the description-kNN feature graph, and the token feature matrix all feed GCN branches whose losses — a shared-parameter consistency loss plus HSIC decorrelation between shared and distinct embeddings — act as regularisers on the jointly trained encoder.

So the mechanism is: *entities that are structurally distinct are pushed apart in the text embedding space, and the text encoder inherits that separation.* This is a real idea and it works: an average boost over SpEL of 16.2% at a one-year gap growing to 20.9% at three years, on continual entities.

Two of TIGER's own figures tell you where the mechanism lives. Figure 6 shows the recall improvement over BLINK rising with node degree. And the paper's own boost table shows the continual-entity gains consistently exceeding the new-entity gains, which the authors attribute to new entities lacking historical context.

---

## 4. Why TIGER still has a problem with emerging entities

Four arguments, in increasing order of force.

### 4.1 The degree argument

Graph-TempEL is built by intersecting TempEL entities with Wikidata5M, a **static July 2019 snapshot**. TIGER restricts itself to 2019–2022 for exactly that reason (to avoid leakage). But the consequence is that an entity born in 2021 cannot have Wikidata5M relations — it did not exist when that KG was frozen. Its structure-graph degree is zero. TIGER's benefit curve is increasing in degree; emerging entities sit at the origin of that curve.

### 4.2 The feature-graph argument (the interesting one)

TIGER's second graph is a kNN graph over BERT embeddings of entity descriptions. Every entity has neighbours here, including new ones — so this branch does cover emerging entities. But consider what its neighbours *are*. For a stub describing the 2021 edition of an annual tournament, the description-nearest entities are the 2020 and 2019 editions. Graph convolution over this neighbourhood is a smoothing operator: it averages the entity's features with its siblings'. **For exactly the entities whose identity lives in the residual difference from their siblings, feature-graph smoothing moves the representation in the wrong direction.** The HSIC distinctness loss operates between the shared and distinct *graph* embeddings, not between an entity and its near-duplicate siblings; it does not fix this.

### 4.3 The temporal-signal argument

Both Graph-TempEL and GCL-TempEL derive per-year graphs from a single Wikidata5M snapshot intersected with yearly TempEL entity sets, and both papers state that the number of entities is the same across all temporal snapshots. That means the "cross-year edge changes" CYCLE trains on are not observations of real KG revision history; they are artefacts of which entities happen to be representable per year. This is worth verifying directly against the released Zenodo data — if it holds, it is a serious methodological finding and a publishable one, and it weakens the temporal interpretation of both models' gains.

### 4.4 The plateau argument

The strongest evidence is empirical and is in the table at the top of this document. CYCLE changes the graph mechanism substantially — graph attention instead of GCN, cross-year contrastive positives and negatives instead of shared/distinct decomposition, and an explicit claim of robustness for low-degree entities. On continual entities it matches TIGER. On new entities at recall@1 it moves from .186/.195/.188/.217 (TIGER) to .184/.191/.192/.222 (CYCLE) across gaps 0–3. **That is noise.** Two different graph mechanisms, one of which was specifically designed for sparse-connectivity entities, land in the same place.

> TIGER is not limited by *which* graph mechanism it uses. It is limited by its dependence on a pre-existing entity representation: the entity is a single point in a space whose geometry was shaped before that entity existed, and the only channel through which the entity can assert its identity is a stub description that it shares almost entirely with its neighbours. Swapping the graph loss does not touch that.

---

## 5. What RAED solves, and what it does not

**Solves.** RAED attacks population (d) directly and well. It drops the assumption of a complete inventory, retrieves general passages from Wikipedia (rather than entity descriptions), and generates a title plus definition for the mention. On TempEL 2020–2022 with a 2019-trained TempEL model, replacing generated-without-retrieval descriptions with RAED descriptions raises accuracy@64 from 51.2/37.8/44.5 to 57.0/47.4/53.5. Its retrieval-training idea — using wikilink-bearing passages as positive retrieval targets rather than entity summaries — is genuinely good and is the part most worth borrowing.

**Its own stated limits.** RAED does not discover emerging entities; it generates a description for a mention it is handed. Its limitations section says integrating discovery is future work. It does not create a persistent, reusable entry.

**Two limits it does not state, which matter for us.**

- **Its disambiguation rule is maximally vulnerable to sibling confusion.** RAED does ED by embedding the generated description with a sentence-transformer and taking the arg-max cosine against candidate reference descriptions. For a near-duplicate family — the 2021 vs 2020 edition of the same event — a fluent, factually grounded generated description will be cosine-close to *every* sibling. Generation fidelity and discriminability are different objectives, and RAED optimises the first.
- **Its TempEL protocol is not strictly temporally sound.** The index is built from a January 2023 Wikipedia dump. RAED excludes passages from the emerging entities' own pages to prevent leakage, which is the right instinct, but *other* 2023 pages contain post-hoc knowledge about those entities. A 2023 page can describe a 2021 event with information nobody had in 2021. This is an opening, not just a criticism: building a temporally-correct retrieval index for TempEL is a concrete contribution nobody has made.

**The ceiling it establishes.** RAED's appendix reports gold Wikipedia descriptions at 71.0/64.8/72.3 against RAED's 57.0/47.4/53.5. There are 14–17 accuracy@64 points of headroom between generated and gold descriptions. Description quality is a real lever and is not exhausted.

---

## 6. What transfers from Apple's on-device recognition system

Treat it as an analogy about estimator structure, not architecture. Four things transfer; two do not, and saying so out loud will protect you in review.

**Transfers**

- **Global metric, local estimate.** The reason face recognition works from a handful of photos is that the *discriminative subspace* is learned once over a large population of identities, and each individual is then a cheap low-sample estimate inside that subspace. The EL analogue is not obvious and is not what current models do: BLINK-style in-batch negatives are a poor approximation of identity-discriminative training over millions of entities.
- **A gallery of exemplars beats a centroid.** An identity is represented by several canonical observations, not one average. The average of several views destroys precisely the information that distinguishes similar faces.
- **Conservative first pass, deferred merge.** High-precision clustering first, aggressive merging later when more evidence has arrived. Decisions that would be wrong now are postponed rather than forced.
- **An explicit unknown bin, and the representation adapts as observations arrive.** Unknown is a first-class outcome, not a thresholding accident.

**Does not transfer**

- Faces have a stable, physically-grounded identity signal with bounded within-identity variation. Entity descriptions do not: the same entity's description at $t$ and $t+3$ can differ arbitrarily, and two distinct entities can have near-identical descriptions. The signal-to-noise structure is inverted.
- A photo gallery is a closed set of tens to hundreds of identities per user. A Wikipedia KB is open, adversarially crowded, and has millions of entries. Clustering heuristics that are safe at gallery scale are not safe here — and this is the main reason a naive "growing entity memory" will produce wrong merges.

---

## 7. Literature gap

Labelled by confidence. "Verified" means read from the primary paper or its abstract; "reported" means taken from a secondary source and still needing a primary check.

| Work | What it establishes | Status |
|---|---|---|
| TempEL (NeurIPS 2022) | The benchmark; degradation asymmetry; the recurrent-event / page-extract composition of new entities; the COVID subset. | verified |
| TIGER (2024) | Graph-shaped training of a text-only-inference bi-encoder. Gains scale with degree. Smaller gains on new entities. | verified |
| **CYCLE (CIKM 2024)** | Same lab, successor to TIGER. Cross-year graph contrastive learning; claims robustness for low-degree entities. **Emerging-entity recall@1 is indistinguishable from TIGER's.** This is the most important paper for your framing and it is absent from the current draft proposal. | verified |
| RAED (EMNLP 2025) | Retrieval-grounded description generation; strong on TempEL emerging splits; no discovery, no persistence. | verified |
| EDIN (EMNLP 2022) | End-to-end unknown entity discovery, clustering, and indexing. Closest prior art to "growing entity memory." RAED states the EDIN benchmark has not been publicly released. | verified |
| BLINKout (CIKM 2023) | NIL as a learned special entity with joint NIL classification; KB pruning and KB versioning as dataset-construction strategies. Reports that improving NIL discovery did not degrade in-KB linking. | verified |
| STAMO (TKDE 2023) | "Learning EL features for emerging entities": self-training to produce prior, relatedness, and embedding features for entities absent from Wikipedia, from a small number of web documents. **Direct prior art for constructing representations for out-of-KB entities.** Also missing from the draft. | verified |
| DELICATE (2025) | Neuro-symbolic diachronic EL: re-ranks bi-encoder candidates with temporal-plausibility and entity-type-consistency features from Wikidata. Historical Italian, long-tail rather than emerging. Closest existing use of *class* information as a temporal EL signal. | verified |
| DynamicER (2024) | Dataset for resolving emerging *mentions* of dynamic entities; complements TempEL by annotating mention variation over time. | reported |
| EL agents (2025–26) | LLM agents for EL on short QA questions; notably prefer lexical (BM25) retrieval because dense retrievers handle entity-centric queries poorly. Not temporal. | verified |
| Lifelong EL (2025) | Sequential learning of new entity *domains* without retraining, via weight pruning. Solves parameter-level forgetting, not entity-level birth. | reported |

### The gap, stated precisely

> The field has two mature answers for emerging entities — sharpen the global embedding geometry (TIGER, CYCLE), or inject external knowledge into the entity description (RAED, STAMO) — and one mature answer for the open-world question (BLINKout, EDIN). Nobody has asked which of these is the right answer *for which emerging entity*, because nobody has decomposed the emerging-entity error into its retrieval, discrimination, knowledge, and calibration components. The result is that the graph line has plateaued while optimising for a failure mode that may not be dominant, and the retrieval line optimises description fidelity when the dominant failure may be that fidelity and discriminability diverge.

---

## 8. Our core research hypothesis

### 8.1 First, the paradigm comparison

| | Paradigm | Assessment |
|---|---|---|
| A | mention → existing entity | Assumes the KB is complete and the entity vector is adequate. Fails on both counts at birth. |
| B | mention → retrieved candidates → selection | The current standard. Improves recall@K but leaves the selection step scoring against the same weak entity vectors. |
| C | mention → evidence → synthesised entity representation → selection | RAED. Correct instinct, wrong objective: it synthesises for *fidelity*. A faithful description of a near-duplicate is still a near-duplicate. |
| D | mention → evidence accumulated over time → dynamic entity → linking | The draft proposal, and EDIN. Its silent assumption is that accumulation is monotonically helpful. Under crowding it is not: averaging more evidence about a sibling-dense entity pulls it *toward* the sibling mean. Also barely testable on TempEL, which gives you ten mentions per entity inside one yearly bucket. |
| E | mention → discovery + induction + graph reasoning | Most components, least principle. This is the "bag of modules" outcome you said you wanted to avoid. |
| **F** | **mention → reference class → discriminative evidence selection → residual representation → factored linking decision** | Proposed below. Differs from C and D in the *objective* of the construction step, and makes the linking decision factorise into a question that is easy (which class?) and a question that is hard (which member?). |

### 8.2 The hypothesis

> **H.** On TempEL's emerging subset, the dominant error is not missing knowledge but *within-reference-class indistinguishability*. An emerging entity should therefore not be represented as an absolute point estimated from its own description. It should be represented as *its reference class's prototype plus a low-dimensional individuating residual*, where the residual is estimated only from evidence that is anomalous with respect to that reference class, and where the amount of trust placed in the residual scales with how much evidence supports it.

The three claims packed into that, each separable and each testable:

1. **Structural.** The link decision factorises as $p(e \mid m) = p(R \mid m) \cdot p(e \mid R, m)$. Current bi-encoders compute a single dot product that is dominated by the first factor, because class-level semantics account for most of the variance in description embeddings. Emerging-entity errors live almost entirely in the second factor.
2. **Estimation.** Building an entity's representation at birth is a low-sample estimation problem. The right estimator shrinks toward the class prototype in proportion to how little individuating evidence exists — not a heuristic, a bias–variance tradeoff with a closed form.
3. **Selection, not accumulation.** Evidence should be weighted by how much it *separates* the entity from its reference class, not by how relevant it is to the mention. Relevance-ranked retrieval systematically returns class-shared content, because that is what the entity and its siblings have most of.

Claim 3 is the sharpest departure and predicts something the accumulation view does not: **the benefit of adding evidence should be non-monotone in reference-class density**. Sparse class, accumulation helps. Dense class, relevance-ranked accumulation plateaus or hurts, while discriminative selection still helps.

---

## 9. The proposed method, conceptually

Working name: **NASCENT** (neighbourhood-anchored shrinkage for contrastive entity representation). Treat the name as a placeholder.

| | Stage | In → out | Why, and which failure mode it addresses |
|---|---|---|---|
| 1 | Input | Mention span, left/right context, snapshot time $t$ → typed query | Same as BLINK, plus an explicit timestamp that gates every downstream information source. |
| 2 | Temporal knowledge at $t$ | $\mathrm{KB}(t)$ inventory and descriptions, KG edges valid at $\le t$, passage index from revisions $\le t$, category/type graph at $t$ | Everything the system may see. Defining this is the protocol contribution (§13). |
| 3 | Mention representation | Context → $f(m)$ | Unchanged from BLINK. Deliberately: we want the contribution to be attributable to the entity side and the scoring function. |
| 4 | Candidate discovery | $f(m)$ → top-K from $\mathrm{KB}(t)$ | Addresses B. Use the existing bi-encoder retriever, plus a lexical (BM25) leg — dense retrievers are documented to handle entity-centric queries poorly, and emerging entities are the extreme case. |
| 5 | Reference-class induction | Top-K + $\mathrm{KB}(t)$ → class $R$, prototype $\mu_R$, scatter $\Sigma_R$ | The pivot. The candidate set already *is* approximately the reference class — the top-K for a mention of the 2021 edition contains the 2019 and 2020 editions. We are reusing information that is currently thrown away. Sharpen with categories/types available at t. |
| 6 | Emerging-entity detection | $p(R \mid m)$, within-class margins → one of three routes | Addresses F, and separates it from E. High class fit + no member with residual support → new member of a known class. Two members with comparable support → genuine ambiguity. No class fits → novel-knowledge case, route to stage 7b. |
| 7a | Residual construction, crowded case | Evidence $\le t$, weighted by class-anomaly → residual $\hat{r}_e$ | Addresses A for population (b). Evidence is scored by how far it projects *outside* the class subspace, not by mention relevance. |
| 7b | Description synthesis, novel-knowledge case | Retrieved passages $\le t$ → grounded description | Addresses D. This is the RAED mechanism, invoked only where the diagnosis says it applies. Not a bolted-on module — the routing decision at stage 6 is what selects it, and that decision falls out of the same factorisation. |
| 8 | Evidence aggregation | $\mu_R$, $\hat{r}_e$, effective count $n_{\text{eff}}$ → $\hat{e}_t$ | Shrinkage, not averaging. With one stub, the estimate is nearly the class prototype — which is the *safe* answer, and correctly expresses "I know what kind of thing this is and not which one." |
| 9 | Linking decision | $f(m)$, $\hat{e}_t$, $\mu_R$, $\Sigma_R$ → score | Two terms: class fit, and a within-class whitened term. The second is where emerging-entity accuracy is won or lost. |
| 10 | Rejection | Per-class calibrated margin → link / abstain / NIL | Addresses F. Crucially *per class*: absolute similarity is not comparable across classes of different density, which is why global thresholds fail. |
| 11 | Update | New confirmed mention → $n_{\text{eff}}$ ↑, shrinkage ↓ | The maturation mechanism, with a closed form rather than a clustering heuristic — and therefore with a quantitative prediction attached (§12). |

---

## 10. Why this could beat TIGER on emerging entities

Not because it has more components. Because it operates on the axis TIGER cannot reach.

- **TIGER's signal is unavailable precisely where we act.** Its structure graph gives an emerging entity degree zero; its feature graph smooths the entity toward its siblings. We use the sibling set as the *contrast*, not as the neighbourhood to average with. Same input, opposite operation.
- **It changes the scoring function, not just the embedding.** Within-class whitening removes the directions along which the reference class carries no discriminative information. A dot product in the raw space cannot express this, no matter how the space was trained.
- **It fixes the negatives.** In-batch random negatives never contain the sibling. Reference-class negatives always do. Note honestly that this alone is standard hard-negative mining — which is why it must be run as a *control*, not claimed as the contribution (§15).
- **It degrades gracefully instead of guessing.** With one stub and no evidence, the estimator returns the class prototype and a wide margin — the model says "an FAI Cup, I can't tell which," which is both correct and useful. The current architecture must emit a confident arg-max.
- **It has a place for RAED's mechanism without being RAED plus TIGER.** The routing decision is derived from the same factorisation as the scoring function, so knowledge injection is what the model does when it has diagnosed a knowledge problem.

**What must be true for any of this to happen** — state this in the paper, and test it first:

1. A large share of emerging-entity errors are within-reference-class confusions.
2. The gold entity is usually present in the top-K (otherwise this is a retrieval problem and representation work is beside the point).
3. Reference classes are inducible at $t$ without leakage.
4. Within-class covariance is estimable — classes must be large enough, or covariance must be pooled across classes.
5. Evidence carrying residual information exists at $\le t$ for a non-trivial fraction of entities.

---

## 11. Why this is actually novel

| Method | Main representation | Candidate source | Handles new entities how | Temporal? | Graph? | Retrieval? | Main limitation for emerging entities |
|---|---|---|---|---|---|---|---|
| BLINK / bi-encoder | Static point from description | Dense top-K over inventory | Zero-shot generalisation only | No | No | No | Entity vector estimated from one stub; random negatives never contain siblings |
| TempEL baseline | Same, per-snapshot finetuned | Same | Snapshot-specific finetuning | Snapshot-level | No | No | Only 1,764 new-entity training mentions per year |
| TIGER | Text vector shaped by GCN losses | Dense top-K | Better global separability | Training-time | Training only | No | Degree-0 at birth; feature-graph smoothing pulls toward siblings |
| CYCLE | Text vector shaped by cross-year GCL | Dense top-K | Contrast on added/removed edges | Training-time | Training only | No | Empirically indistinguishable from TIGER on new entities |
| KG-ZESHEL / graph EL | Text + graph vectors fused | Dense top-K | Zero-shot via graph priors | No | Yes | No | Not temporal; assumes the entity is in the graph |
| ZESHEL-style zero-shot | Cross-encoder over descriptions | Domain inventory | Reads descriptions at inference | No | No | No | Assumes a usable description exists — false at birth |
| BLINKout | Plus a learned NIL entity | Dense top-K + NIL | Detects out-of-KB | Via KB versioning | No | No | Detection only; no representation afterwards |
| EDIN | Cluster-level from mentions | Inventory + discovered | Discover, cluster, index | Partial | No | No | Clusters by similarity — exactly what merges siblings; benchmark not released |
| STAMO | Learned features from web docs | Inventory + EEs | Self-training over labelled docs | No | No | Document set | Needs labelled documents per entity; not temporally scoped |
| RAED | Generated title + definition | Passage index, no inventory | Retrieval-grounded generation | Split-level | No | Yes | Optimises fidelity; arg-max cosine over sibling descriptions; index not temporally sound |
| EL agents | LLM reasoning over retrieved text | Tool-based | Reason over candidates | No | No | Yes | LLM pretraining postdates the test snapshot — contaminated on TempEL |
| Lifelong / continual EL | Pruned-parameter models per domain | Inventory | Sequential domain learning | Sequence-level | No | No | Parameter-level forgetting, not entity birth |
| DELICATE | Bi-encoder + GBT re-ranker | Dense top-K | Type + temporal plausibility features | Yes | Wikidata features | No | Symbolic feature re-ranking; no estimator for an under-observed entity |
| **Proposed** | **Class prototype + shrunk individuating residual** | **Dense + lexical, reused as the class** | **Factored decision; class-anomalous evidence; per-class calibration** | **Strict $\le t$ protocol** | **Optional, as class structure** | **Routed, not default** | **Fails entirely if errors are retrieval-bound rather than discrimination-bound** |

**What is genuinely new, stated conservatively.** Whitening, shrinkage estimators, hard-negative mining, type-based re-ranking, and retrieval-augmented description generation all exist. The new things are: (i) treating the entity representation at birth as an explicit low-sample estimation problem with a shrinkage estimator whose strength is tied to observed evidence, rather than as a fixed encoding; (ii) selecting evidence by class-anomaly instead of by relevance, with the prediction that these diverge exactly where the problem is hardest; (iii) a per-reference-class calibrated rejection rule that separates "unknown entity" from "known but ambiguous," which a global threshold provably cannot do when class density varies. *If the experiments in §14 show the errors are not within-class confusions, none of this is novel because none of it is relevant.*

---

## 12. Mathematical formulation

Four representation regimes, in increasing generality:

$$
\begin{aligned}
&\text{static:} && e = g(d_e) \\
&\text{temporal:} && e_t = g(d_e^t) \\
&\text{evidence-conditioned:} && e_t = g\left(d_e^t, V_t(e)\right) \\
&\text{induced (ours):} && e_t = \mu_{R(e,t)} + \lambda\text{-weighted residual from class-anomalous evidence}
\end{aligned}
$$

**Setup.** $\mathrm{KB}(t)$ is the inventory at $t$; $d_e^t$ is $e$'s description at $t$; $V_t(e)$ is all evidence about $e$ timestamped $\le t$.

**1. Reference class and its statistics**

$$
\begin{aligned}
R(e,t) &= \text{top-}k \text{ nearest entities to } e \text{ in } \mathrm{KB}(t) \setminus \{e\} \\
\mu_R &= \text{mean of } g(d_{e'}^t) \text{ over } e' \in R \\
\Sigma_R &= \text{within-class scatter of the same set}
\end{aligned}
$$

**2. Decomposition**

$$
g(d_e^t) = \mu_R + r_e, \qquad r_e = \text{the individuating residual}
$$

**3. Discriminative evidence weight**

$$
w(v) \propto \exp\left(\frac{\lVert P_R\, \psi(v) \rVert}{\tau}\right), \qquad P_R = \Sigma_R^{-1/2}\left(I - \hat{\mu}_R \hat{\mu}_R^{\top}\right)
$$

- an approximation to $\log p(v \mid e) - \log p(v \mid R)$
- i.e. weight evidence by how surprising it is given the class

**4. Shrinkage estimator**

$$
n_{\text{eff}} = \sum_v w(v), \qquad \hat{e}_t = \mu_R + \frac{n_{\text{eff}}}{n_{\text{eff}} + \lambda}\, \hat{r}_e
$$

- $\hat{r}_e$ = weighted mean residual of the evidence
- $\lambda$ tracks within-class variance: high variance → trust the evidence sooner

**5. Factored score**

$$
s(m, e) = \underbrace{\beta \cdot \langle f(m), \hat{\mu}_R \rangle}_{\text{which class?}}
        + \underbrace{\left(f(m) - \mu_R\right)^{\top} \Sigma_R^{-1} \left(\hat{e}_t - \mu_R\right)}_{\text{which member?}}
$$

**6. Open-world rule, calibrated per class**

$$
\begin{cases}
\text{link} & \text{if } \max_{e \in R} \text{ within-class margin} > q_R(\alpha) \\
\text{NIL} & \text{if class fit high, no member clears } q_R(\alpha) \rightarrow \text{new member of a known class} \\
\text{route} & \text{if no class fits} \rightarrow \text{novel-knowledge case}
\end{cases}
$$

Two consequences worth putting in the paper.

**The shrinkage factor is the maturation curve.** $\frac{n_{\text{eff}}}{n_{\text{eff}} + \lambda}$ gives a parametric prediction for how accuracy should improve as evidence accumulates. That is a falsifiable functional form, not just "more evidence is better." Fit $\lambda$ on validation, check the predicted curve on test.

**The second term is where the contribution lives.** A plain dot product is a special case: set $\Sigma_R = I$, $\mu_R = 0$, $\lambda = 0$. So the baseline is nested inside the model, which makes the ablations clean and makes the claim precise: we are asserting that the restriction the baseline imposes is exactly the one that costs accuracy on emerging entities.

---

## 13. Strict temporal evaluation protocol

This section is probably worth more than the model. Nobody has done it properly and it is the part a careful reviewer will trust.

**Known at $\le t-1$**
- Model parameters (all training data)
- $\mathrm{KB}(t-1)$ inventory and descriptions as of $t-1$
- KG edges with valid-from $\le t-1$
- Passages from page revisions $\le t-1$
- Aliases and redirects existing at $t-1$

**Known at $t$**
- $\mathrm{KB}(t)$ inventory; descriptions *as of $t$*
- The mention's own document
- Other mentions with timestamp $\le$ the current one, if the transductive setting is declared
- Edges and passages with timestamp $\le t$
- Category / type memberships as of $t$

**Forbidden ($> t$)**
- Any later revision of the entity's own page
- Later revisions of *other* pages that mention it
- Edges added after $t$; aliases created after $t$
- Later KB states of any kind
- **Any LLM whose pretraining corpus postdates $t$**

**The contamination rule deserves emphasis.** Every current general-purpose LLM has read Wikipedia well past 2022. Running one on TempEL 2020–2022 and reporting a number is measuring memorisation, not linking. If you use an LLM at all, either use one with a verifiable pretraining cutoff before $t$, or report it explicitly as a contaminated oracle upper bound. Say this in the paper — it is a contribution, because a lot of recent EL work quietly violates it.

### Three protocol decisions you must declare

1. **Inductive vs transductive.** TempEL's test mentions have no intra-year timestamps, so "accumulate evidence from earlier mentions" requires imposing an arrival order. Report a strict inductive baseline (each mention scored alone) alongside any prequential online result, and never report only the latter.
2. **The retrieval index.** TempEL was built from February 2022 Wikipedia history dumps, selecting the most stable December version of each page per year, and the authors released a regeneration framework. That means a per-year passage index is buildable from the same source. Build it. Then re-evaluate RAED-style retrieval under it — if the reported gains shrink, that is a finding.
3. **Granularity.** TempEL's framework explicitly supports regenerating at quarterly rather than yearly spacing. Quarterly snapshots would give real arrival order and make the maturation curve in §12 directly measurable. This is the single highest-value dataset extension available to you and it requires no new annotation.

---

## 14. Experimental design

### 14.1 Run the diagnostic before you build anything

One week of work, no new model. On the emerging subset, for each error made by a reproduced BLINK / TIGER / CYCLE baseline, classify it:

- Gold absent from top-64 → **B**, retrieval failure.
- Gold present; predicted entity is in the gold's reference class → **A/E**, within-class confusion. *This bucket is the hypothesis.*
- Gold present; predicted entity is unrelated → representation failure of a different kind.
- Gold's description at $t$ is a near-empty stub, or the entity is in the novel-knowledge population → **D**.

If the within-class bucket is small, stop and pivot. This is the cheapest decision point in the whole project and it comes first.

### 14.2 Evaluation slices

- **Emerging vs continual** — continual non-regression is mandatory, not optional.
- **Reference-class density** — bucket entities by mean cosine to their top-10 $\mathrm{KB}(t)$ neighbours, computed with a frozen $t-1$ encoder. This is the primary axis and the one the hypothesis makes a directional prediction about.
- **Derivative vs novel-knowledge** — operationalise TempEL's manual categories: does the title contain a year and match an existing series? Is it a substring-extension of an existing title? Cross-check a manual sample.
- **Description richness** — stub vs developed, by token count at $t$.
- **Evidence count** — for the maturation curve.
- **Graph degree at $t$** — to show directly where TIGER's mechanism runs out and yours starts.
- **Temporal distance** — gaps 0–3, forward-only, plus the 2021 COVID subset reported separately.

### 14.3 Metrics, and why each earns its place

- **Accuracy@64** — keeps you comparable with TempEL and RAED. Also your retrieval-failure measure.
- **Recall@1 and MRR** — keeps you comparable with TIGER and CYCLE, and is where discrimination shows up. @64 will hide your effect; both papers note their own gains shrink as K grows.
- **Sibling-confusion rate** — the fraction of errors whose prediction lies in the gold's reference class. This is the mechanism metric. Report it for every system, including baselines. If your method improves accuracy without reducing this, your explanation is wrong even if your number is right.
- **Calibration** — ECE and a risk–coverage curve (AURC). A model that knows when it doesn't know is a real contribution under crowding.
- **Open-world** — NIL precision/recall/F1, and AUROC for in-KB vs out-of-KB, following BLINKout's setup so the numbers mean something to that community.

### 14.4 Baselines you must run yourself

Reproduced, not quoted: BLINK bi-encoder, TempEL's released per-year models, SpEL, TIGER, CYCLE (all have public code), RAED descriptions fed into the TempEL index, and BLINKout for the NIL comparison. Add two controls that exist specifically to shrink your own claim: **whitening-only** (apply $\Sigma_R^{-1}$ to a frozen BLINK, change nothing else) and **sibling-hard-negatives-only** (BLINK trained with reference-class negatives instead of in-batch negatives). If either of those recovers most of your gain, say so — the paper survives, the framing changes.

---

## 15. Ablations

The minimum set that maps one-to-one onto the three claims in §8.2. Resist adding more.

| Ablation | Question it answers |
|---|---|
| Full model | Reference point. |
| − within-class whitening (keep shrinkage and evidence) | Is the factored score the cause, or is it doing nothing? Tests claim 1. |
| − discriminative selection (relevance-ranked evidence instead) | Selection vs accumulation. Tests claim 3. **The single most informative ablation in the paper.** |
| − shrinkage ($\lambda = 0$, pure evidence estimate) | Does the low-sample estimator matter, or would plain averaging do? Tests claim 2. |
| Fixed $\lambda$ vs $\lambda(n_{\text{eff}})$ | Is maturation real, or is a constant interpolation enough? |
| Single prototype vs multiple exemplars | The Apple-derived question. Does a set-valued class representation beat a mean? |
| − routing (always 7a, or always 7b) | Does population-dependent routing beat applying one mechanism everywhere? If not, the paper is simpler and should be rewritten as such. |
| Oracle reference class | Upper bound on the mechanism. Separates "wrong idea" from "right idea, bad class induction." |
| Oracle description (gold, future) | Headroom, comparable to RAED's gold row. Leaks by construction — label it as a ceiling, never as a result. |

---

## 16. Failure cases and falsification

### The killer experiment

**The sibling contrast set.** Build controlled pairs: an emerging entity $e$ born at $t$, and an already-known sibling $s \in R(e)$ — the previous edition of the same event, the parent page a spin-off was extracted from. Take test mentions of both. On this set, the class-level information is held constant by construction; only the residual can decide.

Three things must all hold:

1. Your method improves disproportionately on the contrast set relative to a matched emerging set with no close siblings.
2. A generic retrieval improvement (better recall@64, no change to scoring) does *not* improve on the contrast set.
3. The sibling-confusion rate falls by roughly the amount the accuracy rises.

If all three hold, you are solving emerging-entity linking. If accuracy improves but (2) or (3) fails, you have improved generic entity retrieval and should say so.

### What would falsify the hypothesis

**Falsifier 1 — the diagnostic comes back retrieval-bound.** If most emerging-entity errors are gold-not-in-top-64, this is a candidate generation problem. Representation geometry is irrelevant. Pivot to retrieval, or to RAED-style inventory bypass.

**Falsifier 2 — errors are diffuse, not within-class.** If predicted entities are mostly unrelated to the gold's reference class, there is no residual structure to exploit and the whole factorisation is wrong.

**Falsifier 3 — selection equals relevance.** If the "relevance-ranked evidence" ablation matches the full model, claim 3 is dead and what remains is hard-negative mining with extra steps.

**Falsifier 4 — optimal $\lambda$ is degenerate.** If tuned $\lambda$ goes to $0$ (never shrink) or to infinity (never trust evidence), the estimator story collapses into either plain averaging or a pure class-prior model.

**Falsifier 5 — no evidence exists at $\le t$.** If, under a strictly temporal index, most emerging entities have essentially no pre-$t$ evidence beyond their stub, then nothing can be estimated and the honest conclusion is that a large part of the emerging-entity gap is *irreducible at that timestamp*. This is a real possibility. It is also a genuinely publishable negative result, and a more valuable one than a 2-point gain.

### Failure cases to expect and report

- Wrong reference class induced → whitening in the wrong subspace → worse than baseline. Should be visible as a bimodal error distribution.
- Very small classes → $\Sigma_R$ ill-conditioned. Mitigate by pooling covariance across classes; report the condition number distribution.
- Over-shrinkage on entities that genuinely are novel → they get absorbed into the wrong class. This is the mirror image of EDIN-style over-merging and you should measure it.
- Continual-entity regression from the added class term. Report it every time.

---

## 17. Reviewer objections, and answers

**"This is LDA / embedding whitening applied to BLINK."**
Partly fair, and it is why whitening-only is a mandatory baseline rather than an afterthought. The distinguishing claims are the per-entity class induced *at inference* for entities with zero training data, the anomaly-based evidence weighting, and the evidence-dependent shrinkage schedule. If whitening-only recovers most of the gain, report that and reframe the paper as an analysis of why it works.

**"The gains are small."**
Pre-register the slices. Report bootstrap confidence intervals and per-slice effect sizes rather than one aggregate number. A two-point aggregate gain that is eight points on the high-density slice and zero elsewhere is a stronger scientific result than a uniform three points, because it confirms the mechanism.

**"Temporal leakage."**
§13, plus the released per-year index. Turn the objection into a contribution by re-running prior work under the strict protocol.

**"TempEL's new entities are mostly recurrent events, so this is a narrow problem."**
Agreed, and that is the paper's premise rather than a weakness — we argue the field has been solving the rare case while the common case went unmeasured. The novel-knowledge subset is reported separately and routed to a different mechanism.

**"Why not just use an LLM?"**
Contamination. Any current LLM has read Wikipedia through the test period; its TempEL performance is unmeasurable as linking ability. Include a contaminated-oracle row explicitly labelled as such — it strengthens the paper rather than weakening it.

**"EDIN already does discovery, clustering and indexing."**
True, and the current draft proposal does not adequately acknowledge it. The distinction is the objective: EDIN clusters by similarity, which is the operation that merges siblings. We are asking how to represent a known-to-be-new entity so that it is separable from the entities it most resembles. Note also that RAED reports the EDIN benchmark was not publicly released, which constrains direct comparison — say so rather than claiming superiority.

**"Your TIGER comparison isn't apples-to-apples."**
Rerun TIGER and CYCLE from their released code on your splits. Both provide code and Zenodo data. Quoted numbers from different splits will not survive review.

---

## 18. Final research thesis

> Existing temporal EL methods assume that an emerging entity is a point in entity space that is merely *badly placed*, and that the fix is to shape the space better — with graph structure, with contrastive objectives over KG edges, or with richer generated descriptions.
>
> We instead propose that an emerging entity is an *under-observed* point whose position is genuinely uncertain at birth, that the information identifying it is almost entirely the small residual by which it differs from a densely populated reference class of already-known entities, and that the linking decision should be factored into a class question and a within-class question, with the entity represented as a class prototype plus a shrinkage-weighted individuating residual estimated only from class-anomalous evidence.
>
> This should help emerging entities because their dominant error mode is confusion with their own siblings rather than absence of knowledge, and because every component of the current pipeline — mean-pooled description encoding, in-batch random negatives, description-kNN graph smoothing, relevance-ranked retrieval, and arg-max cosine selection — systematically destroys the residual signal that decides those cases.

And the sentence to put next to it in the paper: *if the error decomposition shows these are retrieval failures rather than sibling confusions, the hypothesis is false and the correct answer is RAED's.* A proposal that cannot fail is not a proposal.

---

## 19. Three backup directions

### Backup 1 — the measurement and protocol paper (lowest risk, do this regardless)

A temporally-sound retrieval index for TempEL, a quarterly regeneration of the benchmark using the authors' own framework, an audit of whether Graph-TempEL's and GCL-TempEL's cross-year edges reflect real KG evolution, a full A–F error decomposition for BLINK / TIGER / CYCLE / RAED on the emerging subset, and a re-evaluation of prior gains under the strict protocol. This is publishable at a resource or analysis track on its own, it is the first stage of the main project anyway, and it is robust to every falsifier above. **Start here in any case.**

### Backup 2 — per-class calibrated open-world EL

Drop the representation story. Ask only: when is an emerging mention's entity actually linkable, and can the system know? Conformal, per-reference-class abstention; separating "unknown entity" from "known but ambiguous"; selective-prediction and risk–coverage metrics; comparison to BLINKout's global mechanism. Modest scope, clean claim, and the density-dependence of the optimal threshold is a result nobody has reported.

### Backup 3 — temporally-scoped sibling-contrastive training

A training-time-only intervention: replace in-batch random negatives with the entity's nearest $\mathrm{KB}(t-1)$ neighbours, scoped by birth time. Extremely cheap, directly tests claim 3, and works with the small BERT configurations TIGER and CYCLE already use (`bert_uncased_L-8_H-512_A-8`, one epoch, a single A100 for the full experiment grid). Hard-negative mining is well known, so this cannot carry a paper alone — but it is the fastest possible probe of the central hypothesis and it is a required control for the main paper anyway. If it works, you learn that early and cheaply.

---

## Three corrections to the current draft proposal

Since you asked for skepticism, these are the things in the existing research-direction note that would not survive a careful reading:

1. **"Emerging entities are considerably harder than continual entities" is only partly right.** True for degradation, and true in TIGER's and CYCLE's setup. False for TempEL's own absolute accuracy@64, where new entities score higher. Fix this before the professor meeting.
2. **CYCLE is missing and it is the most important gap-related paper.** It is the direct successor to TIGER from the same authors, published at CIKM 2024, and its emerging-entity results are flat against TIGER's. Any "TIGER has a weakness" argument has to account for the fact that its own authors already tried to fix it.
3. **STAMO is missing.** "Learning EL features for emerging entities" (TKDE 2023) is prior art for exactly the "build features for an out-of-KB entity" idea. Not fatal — it is not temporally scoped and needs labelled documents per entity — but a reviewer who knows it and does not see it cited will assume you do not know the area.

The draft's core instinct — that detection is not the interesting part, maturation is — is right. The refinement is that maturation by accumulation is the wrong mechanism for the population TempEL actually contains, and that the shape of the evidence, not the amount of it, is what decides these cases.

---

*Research ideation document. Every number quoted is from the cited paper's own reported results; nothing here is an estimate or a prediction of performance. Literature verified through September 2026 — re-check before submission, particularly for work citing TIGER and CYCLE.*
