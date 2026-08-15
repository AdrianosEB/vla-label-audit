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
| Semantic α (MiniLM-L6-v2) | **0.8125**, 95% CI [0.8107, 0.8140] |
| Nominal α (exact string) | 0.0536 |
| Paraphrase gap | 0.7589 |
| Observed disagreement `D_o` | 0.0890 |
| Expected disagreement `D_e` | 0.4749 |
| Unique instruction strings | 64,516 / 125,276 (51.5%) |
| Effective rank of instruction space | 236.8 of 384 dims |

**Per lab** — each with its own episode-clustered bootstrap CI. CLVR +0.7667 [0.7612, 0.7724]
through GuptaLab +0.8867 [0.8813, 0.8920]. **Refined by Stage A verification:** GuptaLab-highest
is robust (P(argmax) ≈ 1.000 in every encoder arm; survives tie-rate, length, and sample-size
confound checks), but CLVR-lowest is **not** — P(CLVR = argmin) is only 0.73 under MiniLM, RAIL's
CI nearly coincides, and the ordering flips under a tie-removal sensitivity check. Cross-lab
claims must be worded at cluster resolution: GuptaLab uniquely on top, {CLVR, RAIL} the low
cluster. "Lowest in all five encoder arms" is not five independent confirmations — the arms share
the same episodes' sampling noise.

**Stage A (encoder robustness) — measured on the full corpus, adversarially verified.** Full
table and verifier verdicts in `results/stage-A.md`; raw output in `data/encoder_robustness.json`.

| Encoder | Semantic α | 95% CI | D_e | Gap |
|---|---|---|---|---|
| MiniLM-L6-v2 (384d) | 0.8125 | [0.8107, 0.8140] | 0.4749 | 0.7589 |
| mpnet-base-v2 (768d) | 0.8218 | [0.8200, 0.8235] | 0.4132 | 0.7682 |
| gte-base (768d) | 0.8095 | [0.8078, 0.8110] | 0.0400 | 0.7559 |
| bge-large-en-v1.5 (1024d) | 0.8165 | [0.8146, 0.8180] | 0.1454 | 0.7628 |
| TF-IDF floor (2442d) | 0.6310 | [0.6285, 0.6334] | 0.8176 | 0.5774 |

- Neural band width 0.0123 → the absolute α ≈ 0.81 is reportable, quoted with the band (not any
  single CI — encoder choice contributes ~4× the sampling uncertainty) and scoped to
  contrastive sentence-encoder families.
- α does not track anisotropy: D_e spans 20× while neural α moves 0.012; fp16 effects < 1e-5.
- Paraphrase gap survives every encoder (0.756–0.768); not a case/punctuation or duplicate-string
  artifact (nominal α ≈ 0.000 on tie-free episodes while semantic α stays 0.79–0.80).
- Identical-string ties are 5.4% of within-episode pairs; removing tied episodes lowers α ~0.018
  uniformly and leaves the band width unchanged.
- **Per-episode worklists are only neighborhood-stable across encoders**: top-200 overlap
  47–57%, tail-only Spearman 0.06–0.31, but 93–99% of any top-200 lies in another encoder's
  top-2000. Suspect lists are a high-disagreement *pool*, not a precise ranking. This is a
  standalone methodological finding (see `results/stage-A.md`), and it applies retroactively
  to the Session-1 `worst_episodes` list in `data/droid_agreement_results.json`, which is
  MiniLM-only: treat it as a pool sample; any released list is to be recomputed by
  rank-averaging across the four cached neural encoders.
- Gate A premise check: **WEAKENED** — the paraphrase-gap thesis holds, but Stage B's
  per-episode-ranking method inherits the instability above; see the revision note in Stage B.

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

## Stage A — encoder robustness  ✅ complete (Gate A: WEAKENED — see state above and `results/stage-A.md`)

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

**Brief revision required by Gate A (premise WEAKENED; approach approved by owner):**
per-episode embedding-disagreement rankings are only neighborhood-stable across encoders
(top-200 overlap 47–57%), and cross-modal text–image spaces are likely less stable than the
same-modality encoders tested. Do **not** draw the suspect list from a single encoder's
ranking: **rank-average the per-episode disagreement across ≥2 vision encoders** (owner chose
rank-averaging over top-K intersection), treat tail *membership* as the operative guarantee,
and budget the manual inspection expecting roughly half of any single-model top-50 to be
encoder-specific noise.

**Dataset: `lerobot/libero`** — 1.94 GB, 1,693 episodes, 273,465 frames, 2 cameras (546,930
images), 40 distinct instructions. Not the 69.86 GB or 34.94 GB variants. **Acquired and verified
2026-08-15**; every count matches exactly. Storage is **AV1 MP4 (LeRobot v3.0)**, not
PNG-in-parquet as previously recorded here — see the correction in `CLAUDE.md`. Decode is cheap
(~2,400 img/s sequential); parquet holds state/action/index only.

**Two structural facts discovered on acquisition, which change what Stage B can measure:**

1. **LIBERO has no multiple annotations per episode** — one instruction per task_index, one
   task_index per episode, no annotator fields anywhere. The planned "run LIBERO's own annotator
   agreement so DROID has a comparator under the identical statistic" is therefore **impossible**,
   not merely inconvenient. Report as impossible; do not substitute a proxy.
2. **Only 40 distinct instructions across 1,693 episodes** (29/44/50 episodes per instruction,
   min/median/max). The language view has at most 40 unique vectors, so per-episode language
   nearest-neighbours are ties among ~29–50 identical strings and a k-NN Jaccard `neighborhood_
   overlap` between visual and language views is largely degenerate as specified. LIBERO is the
   *templated* extreme — the RT-1 pathology, the opposite of DROID's. Stage B's measurements are
   reframed accordingly (below); this deviation from the original brief is deliberate and is
   reported at Gate B.

**Encoder: DINOv2 ViT-S/14** (384-d). Not ViT-B/14 — 16 GB is *unified* memory shared with the
GPU. Estimated 1–2 hours for the full corpus on this machine, once. **Cache embeddings to disk
immediately**; the M2 Pro has no AV1 hardware decode, so decoding is not free and must not
happen twice.

**Build:** a joint index with three aligned views per episode — visual, action, language — with
`dataset`, `episode_id`, `frame_id`, `task_id`, `annotator_id` on every row. Exact search
(`faiss-cpu`, `IndexFlatIP`), not approximate: at this scale brute force is minutes and under a
gigabyte, and exactness removes the "did ANN recall cause that?" objection pre-emptively.

**Measure** — reframed for the 40-instruction structure; functions exist in
`vla_label_audit.crossmodal`. Deviations from the original brief are marked and are reported
at Gate B.

- `effective_rank` / spectral diagnostics on the visual block first — headroom before method.
- **Visual-neighbourhood task purity — the headline of Stage B, replacing the planned
  language-vs-visual `neighborhood_overlap`.** For each episode, the fraction of its k visual
  nearest neighbours sharing its `task_index`, against the chance baseline (~2.6%, group-size
  weighted). This asks the intended question — does the label describe what the camera saw —
  in the only form the language view can support, since 40 unique instruction strings make
  k-NN Jaccard between views a tie-breaking artifact rather than a measurement. Report
  `neighborhood_overlap` as well, explicitly labelled degenerate, so the reason is on record.
- `rank_correlation_across_views` — whole-geometry version. Valid here, with the caveat that
  language pairwise distances take only ~40×40 distinct values.
- `cca_alignment` — rank-limited to 40 by the language side and prone to overfitting at that
  rank; report **cross-validated / held-out** canonical correlations, not in-sample ones.
- `neighborhood_disagreement` → the ranked suspect list, **rank-averaged across DINOv2 and
  CLIP** per the Gate A decision, never taken from one encoder.
- **NEW — synthetic label-swap validation** (approved). LIBERO's instructions are task
  definitions used to generate scripted simulated demos, so its natural label-noise rate is ≈0
  by construction. An audit run on it therefore measures detector **specificity** (how many
  episodes it flags in a corpus that has almost nothing to find), not detection ability. To
  measure **recall**, swap instructions between a held-out sample of episodes across task
  suites, re-run the detector, and report ROC/AUC for recovering the swapped ones, plus
  **precision@50 and precision@200 reported separately** — Gate A established that the tail is
  exactly where ranking instability lives, so a single top-50 number would flatter the method.
  This turns Stage B from a descriptive run into a calibration of the method itself, and it is
  what makes Stage C's noise-injection interpretable.

  **Top limitation, to be stated plainly in `results/stage-B.md` and in any writeup, not
  buried in a limitations section:** the cross-modal detector is calibrated on *synthetic*
  errors in *simulation*. LIBERO has no real annotation noise to find, and planted
  cross-suite instruction swaps are almost certainly easier to detect than the subtle,
  plausible-but-wrong human labels the method is ultimately aimed at. Any AUC or precision
  reported here is an **upper bound** on real-world performance, and must be described as one.

- **Qualitative real-data spot-check — `lerobot/droid_100`** (100 real DROID episodes, time-boxed
  to ~30 min; skipped and recorded as untested if more expensive). Not a ranking study and not
  powered for one: purely a check that the detector behaves sanely on real human annotations
  rather than only on planted swaps. Report what it showed, or report that it was not run.
- **Cannot run:** LIBERO's own annotator agreement. The dataset has one instruction per task
  and no annotator fields, so there is no comparator for DROID's α under the identical
  statistic. Reported as impossible; no proxy substituted.

**Then do the unglamorous thing:** manually inspect the top ~50 suspects and classify them —
wrong label, genuinely rare behaviour, or detector artifact. Report the precision. An
unvalidated ranked list is a claim, not a result.

### Gate B — stop and report

Is the cross-modal alignment strong, weak, or absent? Does the suspect ranking survive human
inspection? ~~What is LIBERO's α next to DROID's 0.8125?~~ — not answerable, LIBERO has no
multiple annotations (see above). Replacement question, which is the more useful one:
**does the embedding-distance label detector actually work?** Report its specificity on
unmodified LIBERO and its ROC/AUC, **precision@50 and precision@200 separately**, against
injected instruction swaps — with the synthetic-errors-in-simulation caveat stated as the
headline limitation, and the `droid_100` real-data spot-check reported or marked untested.

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
