# Roadmap

Read `CLAUDE.md` first — it holds the standing rules, the machine constraints, and the design
decisions that must not be casually undone. This file holds **state** and **sequence**.

Work through the stages **in order, stopping at each gate.** Every gate exists because the
result of that stage determines what the next one should be. Two hypotheses have already died
in this project (see below); a plan written far enough ahead to ignore that would be encoding
guesses as instructions.

---

## Established state

Everything below is measured, on the full DROID annotation file (50,092 episodes, 125,276
annotations, of which 37,592 episodes are triple-annotated and 12,500 are single).

**Headline**

| Quantity | Value |
|---|---|
| Semantic α (MiniLM-L6-v2) | **0.8125**, 95% CI [0.8109, 0.8140] |
| Nominal α (exact string) | 0.0536 |
| Paraphrase gap | 0.7589 |
| Observed disagreement `D_o` | 0.0890 |
| Expected disagreement `D_e` | 0.4749 |
| Unique instruction strings | 64,516 / 125,276 (51.5%) |
| Effective rank of instruction space | 236.8 of 384 dims |

**Per lab** — each with its own episode-clustered bootstrap CI. CLVR +0.7667 [0.7612, 0.7724]
through GuptaLab +0.8867 [0.8813, 0.8920]; the extremes do not overlap. That pair was selected
post hoc from 10 labs and is **not** a multiplicity-corrected test — suggestive only.

**Defect classes and their cost to α**

| Class | n | α excluding | Δ |
|---|---|---|---|
| truncated | 17 | 0.8126 | +0.0002 |
| junk | 5 | 0.8126 | +0.0001 |
| non-answer | 389 | 0.8115 | −0.0010 |
| no terminal punctuation | 113,641 | 0.7914 | −0.0211 |

The first three sit far inside the baseline CI: **the headline survives cleaning.** The fourth
is not a defect class — 90.7% of annotations lack terminal punctuation, so that is house style,
and excluding it drops 91% of the corpus. Do not report it as comparable to the others.

**Findings that are not just numbers**

- **Degenerate agreement.** 103 of the 115 episodes containing a non-answer have *all three*
  annotators writing "No action". These are perfectly-agreeing units with no instruction to
  agree about, and they inflate α. An agreement statistic cannot distinguish unanimous-correct
  from unanimous-empty. Decision taken: **keep them in the headline, report the exclusion as a
  robustness row, and release the episode list.** They are real episodes that people train on;
  silently dropping them would measure a corpus nobody uses.
- **The file is sorted** — all 37,592 triple-annotated episodes first, then all 12,500 singles.
  `--limit N` is therefore never a random sample and its output must never be quoted as an
  estimate of the full-file number.

**Hypotheses tested and rejected — do not resurrect without new evidence**

- *"Truncated annotations are an annotation-tool artifact (character cap or timeout)."*
  **Rejected.** 17 truncations in 125,276 (0.014%), with 14 distinct lengths among 17 events —
  a fixed cap produces one length. No association with lab (χ²=10.00, p=0.53), month (p=0.61),
  or sibling annotation length (Mann-Whitney p=0.59). Note the χ² tests are underpowered at this
  base rate, so this is "no evidence for", not "evidence against".
- *"Robot language labels are broadly unreliable, and that explains why VLAs ignore language."*
  **Not supported.** α = 0.81 clears the conventional reliability threshold. The surviving,
  better-supported thesis is about the **paraphrase gap** — annotators overwhelmingly mean the
  same thing (0.81) and almost never say it the same way (0.05). Contrast with RT-1, which has
  <2% unique instructions and 49 unique words across 3.7M sentences. Two opposite pathologies:
  RT-1 too templated to teach language, DROID too varied to give repeated signal per phrasing.

---

## Stage A — encoder robustness  ⬅ current

**Full brief: `NEXT_TASK.md`.** Summary: every number above lives in MiniLM's geometry. Re-run
across 4–5 genuinely different encoders plus a **TF-IDF floor**, and measure whether the
absolute α is stable, whether the per-lab *ranking* is stable, and whether the encoders agree
about which specific episodes are worst.

### Gate A — stop and report

Answer three questions before touching Stage B:

1. **Is the absolute α reportable, or only comparisons?** If the encoders span a narrow band,
   report the number. If they scatter widely, the paper becomes strictly comparative.
2. **Is the per-lab ranking stable across encoders?** Rank correlation. This decides whether
   cross-dataset and cross-lab claims are safe even when absolute values are not.
3. **Where does TF-IDF land?** If plain lexical overlap reaches ~0.75, then most of what we call
   semantic agreement is annotators reusing nouns, and the story changes materially. This is the
   single most informative number in the stage.

**Do not start Stage B until Gate A is answered.** Stage B costs hours of image embedding, and
if the embedding-distance approach is shakier than it looks, that has to be known first.

---

## Stage B — LIBERO cross-modal audit

*Only after Gate A.* Goal: move from "do annotators agree with each other" to "does the label
describe the trajectory".

**Dataset: `lerobot/libero`** — 1.94 GB, 1,693 episodes, 273,465 frames, 2 cameras (546,930
images), 40 distinct instructions. Not the 69.86 GB or 34.94 GB variants; identical content,
PNG-in-parquet instead of AV1 video.

**Encoder: DINOv2 ViT-S/14** (384-d). Not ViT-B/14 — 16 GB is *unified* memory shared with the
GPU. Estimated 1–2 hours for the full corpus on this machine, once. **Cache embeddings to disk
immediately**; the M2 Pro has no AV1 hardware decode, so decoding is not free and must not
happen twice.

**Build:** a joint index with three aligned views per episode — visual, action, language — with
`dataset`, `episode_id`, `frame_id`, `task_id`, `annotator_id` on every row. Exact search
(`faiss-cpu`, `IndexFlatIP`), not approximate: at this scale brute force is minutes and under a
gigabyte, and exactness removes the "did ANN recall cause that?" objection pre-emptively.

**Measure** (functions already exist in `vla_label_audit.crossmodal`):

- `compressibility`-style diagnostics on the visual token block first — headroom before method.
- `neighborhood_overlap` between visual and language views. **This is the headline of Stage B.**
  If an episode's visual neighbours are not its language neighbours, the labels are not
  describing what the camera saw.
- `rank_correlation_across_views` — the same question over the whole geometry, not just top-k.
- `cca_alignment` — is there *any* linear map aligning the two? Canonical correlations near zero
  is a much stronger negative than low local overlap.
- `neighborhood_disagreement` — the ranked suspect list.
- Run LIBERO's own annotator agreement too, if it has multiple annotations, so DROID has a
  comparator under the identical statistic.

**Then do the unglamorous thing:** manually inspect the top ~50 suspects and classify them —
wrong label, genuinely rare behaviour, or detector artifact. Report the precision. An
unvalidated ranked list is a claim, not a result.

### Gate B — stop and report

Is the cross-modal alignment strong, weak, or absent? Does the suspect ranking survive human
inspection? What is LIBERO's α next to DROID's 0.8125?

---

## Stage C — what does label noise cost?

*Only after Gate B.* Descriptive results become a causal claim here.

There is no clean version of DROID to compare against, so run it backwards: inject **known**
noise, train at each level, fit the degradation curve, then use the curve to convert a measured
noise rate into a predicted performance cost.

**Policy: a state-based MLP behaviour-cloning policy on PushT.** Roughly 2 minutes per 20k-step
run. **Never ACT, Diffusion Policy, or SmolVLA fine-tuning** — 13+ hours per run on Apple
Silicon; a seeded comparison would take a week. This constraint is what makes the stage possible
at all.

- Budget: ~10 noise conditions × 5 seeds × 2 arms ≈ 100 runs ≈ 3–4 hours.
- Add a small CNN on 96×96 pixels for the headline conditions overnight — the curation operates
  in pixel-embedding space, so a pixel-input policy is the more honest test.
- **~1,500 rollouts per arm.** Gives ±2.5pp intervals; a casual 50-rollout eval gives ±13.6pp
  and cannot detect anything that matters. Rollouts are nearly free and training is not, so
  spend compute there.
- **Report seed-level variance, not just rollout-level.** It will dominate binomial noise and it
  is the number that makes the result credible.
- Install trap: pin `pymunk<7.0.0` or gym-pusht fails with
  `'Space' object has no attribute 'add_collision_handler'`.

**Own the scale mismatch in the abstract**: audited at 273k frames, causality shown at 25k, on a
dataset with one language instruction. That is a defensible design given the hardware; blurring
it is what a reviewer catches.

### Gate C — stop and report

What is the slope? Does the predicted cost at DROID's measured defect rate explain any
meaningful share of the RoboSemanticBench gap, or is it negligible?

---

## Stage D — write-up

*Only after Gate C.* Six to eight pages. Provisional structure, subject to what Stages A–C
actually produced:

1. The measurement nobody made — α with intervals, per lab, across encoders
2. The paraphrase gap, and the RT-1 contrast (two opposite failure modes)
3. Cross-modal alignment
4. What the noise costs
5. Degenerate agreement as a methodological caution
6. Limitations, stated first and aggressively

Then arXiv, then a workshop. Re-run the novelty search before submitting — this area moves
monthly.

---

## Standing rules

Repeated from `CLAUDE.md` because they matter most when a stage is going badly:

- **Never weaken a test to make it pass.** The tests are correctness claims.
- **Never fabricate a number or a citation.** A plausible invented figure in a statistics paper
  is worse than no figure, because nobody can catch it by looking.
- **If you change a statistical formula, prove it** against a naive implementation or an
  analytic result.
- **Ask on research judgement.** "Is this metric right", "is this novel", "does this support the
  claim" are not coding questions.
- **Report what ran.** A failed run is a finding. A hypothesis that dies is a finding — two have
  already, and the project is better for both.
