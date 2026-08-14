# The project, fused with your existing research line

Adrianos Botsios — drafted August 2026

---

## 0. What changed, and why I'd switch

Your VLA line is about **data labeling and curation**. The token-pruning project from earlier
is about **inference efficiency**. They're both VLA work, but they're different research
directions, and putting both on a resume reads as two hobbies rather than one line of inquiry.

You asked to fuse them. The honest answer is you can't — so pick the one that continues what
you're already leading. That's this one. It turns a vague bullet ("researching scalable data
labeling and curation strategies") into a specific finding with a number attached.

Nothing is wasted: `stats.py` from the other repo — clustered bootstrap, Wilson, power
analysis, Holm — transfers here unchanged. It's the same statistical spine.

---

## 1. The finding to aim at

> **The field blames the architecture for VLA models ignoring language. Show the labels are
> broken.**

That's the paper. One causal claim, and every piece needed for it already exists separately
without anyone having connected them.

**The model-side failure is established and quantified.**
[RoboSemanticBench](https://arxiv.org/html/2606.02277) finds VLAs grasping successfully
80–100% of the time while completing the *commanded* task only 2–21% of the time — normalized
semantic grounding at or below zero, meaning near-random target selection once you control for
grasping. Everyone treats this as an architecture problem.

**The data-side cause is completely unmeasured.**
[Wanna et al. (arXiv:2601.03136)](https://arxiv.org/html/2601.03136v1) measured the *text* of
embodied-AI datasets: under 2% of instructions are unique; **RT-1 uses 49 unique words across
3.7M+ sentences**. They then state that cross-modal alignment and annotation correctness are
outside their scope, and name detecting "inconsistencies between commands and the
corresponding trajectories" as future work. That is your project statement, written by someone
else as a limitation.

**The measuring instrument is free and already public.**
[DROID](https://droid-dataset.github.io/) collected **three independent crowdsourced
instructions for 95% of successful episodes** — about 75,000 triple-annotated trajectories —
and published no inter-annotator agreement, no error rate, no language quality validation of
any kind. Nobody has run the numbers on data that has been sitting there since 2024.

Working title: **"Do Robot Datasets Say What They Think They Say? An Audit of Language
Annotation Quality in Robot Learning."**

## 2. Why not the other obvious versions — I checked, and they're taken

I had this adversarially scanned. Three of the four things you'd naturally propose are gone:

| Component | Status |
|---|---|
| Semantic dedup / redundancy | **Taken.** [SCIZOR](https://arxiv.org/html/2505.22626v2) is SemDeDup for robot data, including the joint state-action twist you'd have found in month three. Removes 29.6% of RoboMimic Multi-Human, 15.8% of OXE Magic Soup; +16.1% over uniform random deletion |
| Coreset / data selection | **Dead.** SCIZOR, [SIEVE](https://arxiv.org/html/2607.06442), [DataMIL](https://arxiv.org/html/2505.09603v2), [Re-Mix](https://arxiv.org/abs/2408.14037), [DemInf](https://arxiv.org/html/2502.08623). Random-subset baselines are *standard* here — "we beat random" is table stakes. And [Belkhale et al. (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/fe692980c5d9732cf153ce27947653a7-Paper-Conference.pdf) already showed embedding-diversity heuristics aren't reliably correlated with success |
| Retrieval over demonstrations | **Closed as research** (VINN → BehaviorRetrieval → FlowRetrieval → [STRAP](https://arxiv.org/html/2412.15182v2)). No public vector index exists, so it's an artifact gap — use it as your *tool*, not your contribution |
| **Language annotation quality** | **OPEN.** Nothing published. Searched from six angles |

Also worth knowing: the field agrees this is unsolved. Moritz Reuss's ICLR 2026 VLA roundup:
*"It's an open secret that OXE is mostly low-quality data, yet we still lack good methods to
quantify data quality in imitation learning"* — and *"surprisingly few ICLR 2026 submissions
focused on data collection and curation."*

**Read before you commit:** there's an OpenReview paper titled *"On Data Redundancy in VLA
Training"* that returned 403 on every access route tried. Given the title, assume it may
contain the redundancy numbers. It shouldn't touch the label-quality angle, but verify.

## 3. Your actual question: a vector database of *what*

One index. **Three aligned views per episode**, because every audit question is a cross-view
query.

| View | What goes in | Encoder | Dim |
|---|---|---|---|
| **Visual** | Per-frame embedding, plus a pooled per-episode vector | DINOv2 ViT-S/14 | 384 |
| **Action** | Normalized action chunk, summarized to fixed length | direct / small PCA | ~64 |
| **Language** | The instruction string — *one row per annotator*, not per episode | sentence encoder | 384 |

Metadata on every row: `dataset`, `episode_id`, `frame_id`, `task_id`, `annotator_id`.

That last detail is the one that matters. **Store each annotator's instruction as a separate
row**, not a merged episode label. The whole DROID agreement study lives in that decision —
merge them and you throw away the disagreement signal that makes this project possible.

What the index buys you, concretely:

- *"Which episodes have labels unlike their behavioural neighbours' labels?"* → query the
  visual view, compare the language view. Ranked suspect list, no ground truth needed.
- *"Are visual neighbours language neighbours at all?"* → overlap between two neighbour sets.
- *"Which annotator is an outlier?"* → group by `annotator_id`, compare per-annotator
  disagreement distributions. Nobody has ever looked at this.
- *"How many distinct behaviours does this corpus really contain?"* → effective rank.

**Use FAISS `IndexFlatIP` — exact, not approximate.** At your scale that isn't a compromise:
547k × 384-d is **0.84 GB of RAM and about 38 minutes** for the full self-kNN on a laptop.
Exact search removes a whole class of "did your ANN recall cause that result?" objections
before anyone raises them. `faiss-cpu` ships an arm64 wheel now (macOS ≥14, Python ≥3.10); the
old "FAISS doesn't work on M1" complaints are stale. Avoid `hnswlib` — sdist only, compiles
from source; use `usearch` if you want an alternative.

## 4. Your other question: which mathematical models

Every one of these does real work. None is decoration.

**1. Krippendorff's alpha with an embedding distance — the methodological contribution.**
Standard agreement statistics assume categorical labels. These are sentences: "pick up the red
mug" and "grab the mug" agree, "move the arm left" doesn't, and no categorical statistic sees
the difference. Alpha is defined over an *arbitrary* difference function, so substituting
cosine distance between sentence embeddings makes it measure whether annotators described the
same *behaviour* rather than typed the same *string*. Report semantic alpha next to nominal
alpha — **the gap between them is the paraphrase rate**, and separating paraphrase from real
disagreement is the crux of the whole audit.

**2. Confident learning / neighbourhood disagreement — the label-error detector.** Northcutt
et al.'s insight: you can infer label errors from the structure of the data without a clean
reference set. Transplanted here: an episode whose behavioural neighbours all carry a
different label is a suspect. In the synthetic demo this hits **100% precision in the top 10%
and 4.33× lift over random** at recovering planted errors.

**3. CCA — the corpus-level alignment test.** Asks whether *any* linear map makes vision and
language correspond. Canonical correlations near zero is a far stronger negative result than
low local overlap. And CCA is invariant to invertible linear reparameterization, so one
encoder scaling its outputs differently can't fake alignment — the repo tests exactly this
property.

**4. Mutual information.** Two estimators, deliberately. Gaussian-from-CCA
(`I = -½Σlog(1-ρᵢ²)`) is stable in high dimension and wrong in a known direction. The KSG
k-NN estimator is assumption-free and badly biased above ~10 dimensions. **PCA-reduce before
KSG, use Gaussian as the headline, KSG as the sanity check — never the reverse.** Knowing
which estimator to trust where is the kind of thing that reads as competence.

**5. PCA / effective rank — the deflation number.** A corpus advertising 160,000 tasks whose
instruction embeddings span 8 directions doesn't have 160,000 tasks. It has 8 templates and a
lot of paraphrase. This is the single most quotable number the project can produce, and it's
your HarvardX PCA doing load-bearing work.

**6. Spearman rank correlation on pairwise distances.** Neighbourhood overlap only sees the
top-k; this sees the whole geometry. Near zero means the spaces are unrelated at every scale.

**7. Episode-clustered bootstrap, Wilson intervals, minimum detectable effect, Holm.** Your
signature, and the thing this literature never does. Frames within an episode are heavily
correlated, so an i.i.d. bootstrap understates the spread — resample whole episodes. Annotator
comparisons are naturally paired. Report your detection threshold before anyone asks.

**8. Linear degradation fit.** Injected-noise-rate → performance, fit a slope, then multiply
by the measured real noise rate to get a predicted cost. Deliberately linear: with 5–6 noise
levels and real seed variance, anything richer fits noise.

## 5. The build, on your laptop

### Weekend 1 — the hook, essentially zero compute

DROID's language annotations are **text**. You don't need the 1.7 TB of video to compute
annotator agreement — you need three strings per episode, which is a small file. Embed them
with any sentence encoder, run `krippendorff_alpha` twice (semantic and nominal), bootstrap
the CI, and rank the worst episodes.

**That produces a novel measured number in a weekend**, on a dataset the whole field uses,
that nobody has published. It's the hook for everything else. Report it with a confidence
interval and you're already ahead of the practice in the area.

### Week 2 — the cross-modal audit at scale

Use **`lerobot/libero`**: 1,693 episodes, **273,465 frames, 2 cameras = 546,930 images, 40
distinct instructions — in 1.94 GB**. Not `HuggingFaceVLA/libero` (69.86 GB) or
`physical-intelligence/libero` (34.94 GB) — byte-identical content, 36× larger, because the
LeRobot one is AV1 video rather than PNG-in-parquet.

Embed with DINOv2 ViT-S/14 (12.3 GFLOPs/image, 384-d). Estimated 1–5 hours for the full
corpus, once, cached forever. **Cache embeddings to disk immediately** — on M1/M2 there's no
AV1 hardware decode (Apple added it in M3), so decoding may dominate the pass and you never
want to pay it twice.

Then: `neighborhood_overlap`, `cca_alignment`, `effective_rank`, and the ranked suspect list.

### Weeks 3–4 — the causal claim

This is where the project dies if you pick the wrong policy. **Do not use ACT or Diffusion
Policy.** A LeRobot user on an M1 Max reports ACT training "going to take >13 hours"
([issue #3191](https://github.com/huggingface/lerobot/issues/3191)). Curated-vs-full with 5
seeds each is 10 overnight runs — six days of laptop uptime for one data point with no error
bars. SmolVLA fine-tuning is worse: the documented recipe is ~4h on an A100, and MPS is fp32
with no AMP, so estimate 20–40h.

**Use a state-based MLP behaviour-cloning policy on PushT.** Measured: **~2 minutes per 20k-step
run** on hardware far worse than your laptop. That makes the experiment possible:

- 10 noise conditions × 5 seeds × 2 arms = **100 runs ≈ 3–4 hours**
- Add a small CNN on 96×96 pixels for your headline conditions overnight — worth doing, since
  your curation operates in pixel-embedding space and a pixel-input policy is the honest test
- **Evaluation is nearly free**: `gym-pusht` is pure Python (no MuJoCo), 300-step episodes at
  ~1,490 env-steps/s. **1,500 rollouts per arm ≈ 1 hour** and gives ±2.5pp confidence
  intervals. A casual 50-rollout eval gives ±13.6pp and cannot detect anything you care about

Spend compute on rollouts and seeds, not on a bigger policy. And **report seed-level variance,
not just rollout-level** — with training this cheap you can afford 5+ seeds, and the
between-seed spread will dominate binomial noise. That's the number that makes it credible.

Two install traps, both confirmed: pin `pymunk<7.0.0` or gym-pusht breaks with
`'Space' object has no attribute 'add_collision_handler'`. And **sanity-check any MPS result
against CPU** — [lerobot#496](https://github.com/huggingface/lerobot/issues/496) was a silent
MPS bug where `.to(device, non_blocking=True)` returned garbage and eval scored 0%. It's fixed,
but that class of bug fails silently and you'd have no way to tell a broken backend from a bad
policy.

### The weakness to own, not hide

You audit at 273k frames and prove causality at 25k, on a dataset with **one** language
instruction. State that plainly in the abstract. It's a defensible design given the
constraint; blurring it is what a reviewer catches.

## 6. Risks

| Risk | Response |
|---|---|
| "On Data Redundancy in VLA Training" scoops part of it | Read it first. Shouldn't touch label quality, but check |
| [NILS](https://robottasklabeling.github.io/) already reports an error rate vs BridgeV2 human labels | Unverified — read the full PDF. If it does, it partially erodes the gap and you cite it as motivation instead |
| Someone runs the DROID agreement analysis first | It's a weekend. Do it now and post it |
| LIBERO closed-loop eval on macOS | Rendering path unverified on Apple Silicon. Don't build a headline result on it — PushT is the fallback |
| Alpha comes out high (labels are fine) | Then the architecture explanation stands and you've established that, with the first agreement number in the field. Still publishable, still the right question |

## 7. Two housekeeping notes

**Your HarvardX repo is in R.** Keep it — it's evidence of a statistics background that almost
nobody applying to these roles has. But write this project in Python. Python is the only
universal hard requirement across the postings I scanned (~15 of 21); Anthropic's Fellows
Program lists "fluent in Python" as its *sole* hard requirement. R signals statistician,
Python signals ML engineer, and you want to be read as both.

**Your current bullet is vague and slightly off-key:**

> Researching scalable data labeling and curation strategies for training vision-language-action
> (VLA) models in robotics applications, with a focus on deployment and venture opportunities

"With a focus on deployment and venture opportunities" reads as a business interest inside a
research bullet — for research-lab applications, it undercuts you. Once this project has real
numbers, that bullet becomes something like:

> Built the first inter-annotator agreement analysis of language labels in large-scale robot
> datasets, measuring semantic agreement across [N] independently annotated DROID episodes with
> episode-clustered bootstrap intervals; released an open-source audit toolkit

Fill in the numbers once you have them. Don't write it before.

## 8. What's already built

In this repo, 34 passing tests:

- **`agreement.py`** — Krippendorff's alpha over arbitrary difference functions, semantic and
  nominal distance matrices, Fleiss' kappa (pinned to the published worked example, κ=0.2099),
  unit-clustered bootstrap CI, per-episode disagreement ranking
- **`crossmodal.py`** — exact k-NN, neighbourhood disagreement, neighbourhood overlap,
  distance rank correlation, CCA with the invariance property tested, Gaussian and KSG mutual
  information (KSG validated against the analytic bivariate-Gaussian answer), effective rank,
  instruction-space report
- **`noise.py`** — three noise modes with realised-rate accounting, degradation curve, cost
  prediction
- **`scripts/demo_synthetic.py`** — end-to-end on a corpus with a *known* 18.7% of labels
  corrupted. Recovers it: semantic alpha 0.9991 vs string alpha 0.0000 (paraphrase isolated),
  alignment MI dropping 16.80 → 3.57 nats under the planted noise, and the suspect ranking
  hitting **100% precision at top-10%, 4.33× lift over random**

What isn't done: the real data. That's your weekend, and it's the part that has to be yours.
