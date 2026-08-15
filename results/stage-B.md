# Stage B — LIBERO cross-modal audit

**Top limitation, stated first because it conditions everything below.** The cross-modal
detector evaluated here was calibrated on **synthetic errors in simulation**. LIBERO's
instructions are the task definitions that generated its scripted demonstrations, so its
natural annotation-noise rate is ≈0 by construction: there is nothing real to find. Every
detection number below comes from errors we planted ourselves, and planted cross-task
instruction swaps are far easier to detect than the subtle, plausible-but-wrong human labels
this method is ultimately aimed at. **Any AUC or precision here is an upper bound on
real-world performance and must be described as one.** Adversarial verification then showed
that even that upper bound does not license the claim the stage set out to make — see
"What did not survive".

Corpus: `lerobot/libero`, 1,693 episodes, 273,465 frames, 2 cameras (546,930 images),
40 instructions, simulated. Encoders: DINOv2 ViT-S/14 (primary, self-supervised) and
CLIP ViT-B/32 (ensemble partner, language-supervised — partly circular as an instrument for
judging image–label match, which is why DINOv2 leads). Scripts: `scripts/libero_embed.py`,
`scripts/libero_crossmodal.py`. Raw numbers: `data/libero_crossmodal.json`. Test suite 50/50
throughout. Three adversarial verifiers ran (measurement-triviality; calibration validity;
independent reproduction), each instructed to refute and to default to refuted when uncertain.

## What was measured

| Quantity | Value |
|---|---|
| Visual k-NN task purity, DINOv2 concat (k=1/5/10/25) | 0.9976 / 0.9953 / 0.9917 / 0.9763 |
| Visual k-NN task purity, CLIP concat | 0.9947 / 0.9944 / 0.9916 / 0.9789 |
| Chance baseline `Σ n_t(n_t−1) / N(N−1)` | 0.024809 |
| Label-swap recovery, ensemble, S=85 (5%), 10 seeds | AUC 0.9975±0.0021, P@50 1.000±0.000, P@200 0.4210±0.0032, recall@200 0.9906 |
| Label-swap recovery, ensemble, S=169 (10%), 10 seeds | AUC 0.9965±0.0037, P@50 1.000±0.000, P@200 0.8310±0.0094, recall@200 0.9834 |
| Episodes with any nonzero detector score (unmodified) | **98 of 1,693**; 1,595 exactly tied at zero |
| Episodes flagged by **both** encoders | **23** |
| DINOv2-vs-CLIP tail agreement | Spearman 0.361; top-50 overlap 0.34; top-200 overlap 0.20 |
| Held-out CCA, vision vs language (10 half-splits) | 0.9925 (in-sample 0.9952; permutation control 0.025) |
| `neighborhood_overlap` vision vs language, k=10 | 0.143 mean Jaccard — **degenerate by design**, see below |

P@200 ceilings are arithmetic: S/200 = 0.425 and 0.845, so both conditions sit at ~99% of
the attainable maximum.

## What survived verification

- **All arithmetic.** Independent reproduction rebuilt the per-episode views *bit-exactly*
  from the raw frame embeddings, confirmed the row index tiles all 273,465 rows with no gaps
  or overlaps, and reproduced task purity, the chance baseline, the nonzero counts
  (57 DINOv2 / 64 CLIP / 98 union / 1,595 zero), the tail statistics, and the label-swap
  metrics under its own swap implementation and its own tie-corrected AUC. The suspects CSV
  agrees row for row.
- **Tie handling is correct.** AUC uses mid-ranks (`roc_auc_score` matches a from-scratch
  Mann-Whitney computation to machine precision), and precision@50 is *invariant* to
  tie-breaking: an adversarial tie-break that pushes every swapped episode to the bottom of
  its tie block still yields P@50 = 1.000, because 62–81 swapped episodes score strictly
  above every unswapped one.
- **The CCA held-out path is leak-free.** Weights and means are fitted on the train half only
  and applied to a disjoint test half; a permutation control collapses held-out correlation
  from 0.9925 to 0.025. (Its *interpretation* did not survive — see below.)
- **A real trajectory-dependent signal exists.** The "it's just static scene recognition"
  attack failed on evidence: frame 0 alone gives purity 0.681, not 0.99. In the LIBERO-Goal
  suite, where all 10 tasks share a visually indistinguishable initial scene (frame-0 purity
  0.108 against a within-suite chance of 0.100), full-episode purity is 1.000, and 0.956
  using only the first 75% of each episode. Episode-level leakage was also ruled out: purity
  is 0.9025 after deleting each episode's 20 most-similar same-task rollouts, and 0.9976
  across a disjoint half-split.
- **A genuine, if modest, language-touching result.** The wrong-task neighbours that do occur
  land on semantically nearby instructions: MiniLM cosine 0.771 for confused task pairs
  versus 0.444 for random task pairs (0.552 for same-suite pairs). This is the one statistic
  in the stage that actually reads the instruction text and comes out positive.
- **Two measurement artifacts were caught and fixed before they inflated the headline.**
  (i) Cosines between identical instruction vectors returned ±5.55e-16 rather than 0, and
  that residue is a deterministic function of the vector — so it ordered the 1,595 no-signal
  episodes *identically across encoders* (Spearman exactly 1.0 on that block). (ii) Both
  encoders originally shared a tie-break seed. Left in, the two together reported top-200
  cross-encoder overlap of 0.805 instead of the true 0.20. Snapping at 1e-12 is safe by a
  ~9e8× margin against the smallest possible genuine score (4.53e-4).

## What did not survive

- **"Task purity 40× chance shows the label describes what the camera saw." REFUTED.**
  Two independent grounds, both demonstrated rather than asserted:
  1. **Purity never reads the language.** It is computed from the integer `task_index`.
     Recomputing it under a random permutation of the 40 task ids, and again after replacing
     all 40 instructions with nonsense strings, returns **bit-identical** values
     (0.997637330183107 / 0.9952746603662138 / 0.9917306556408741). A statistic provably
     unchanged when every label is replaced by gibberish cannot support a claim about labels
     describing anything. It measures only that the *partition* induced by `task_index` is
     recoverable from vision.
  2. **The encoders contribute essentially nothing, and the null is too weak.** A per-episode
     mean of an **8×8 RGB thumbnail** — 192 numbers, no neural network — reaches
     0.9988/0.9916/0.9807, matching or beating DINOv2. Against chance 0.0248 the ratio is
     bounded above by 40.31, so "≈40× chance" is the arithmetic ceiling, attainable by
     anything scoring ≥0.987, and carries no information beyond "purity ≈ 1". The result is a
     fact about how visually stereotyped a 40-configuration simulator is, not about
     representation quality or cross-modal alignment.
- **"Held-out CCA ≈ 0.99 shows rich vision–language alignment." REFUTED as interpretation.**
  The procedure is sound, but vision against a plain 40-dim one-hot task indicator gives
  0.9936 held-out — so the canonical correlation is the same fact as task purity restated,
  not evidence of richer structure.
- **"The swap calibration validates embedding-distance auditing." REFUTED in scope.** The
  arithmetic is right; the inference is not. A trivial rule — flag any episode whose
  `task_index` differs from its visual neighbours', using **no language embedding and no
  cosine distance in the label view** — achieves **AUC 1.0000**, strictly beating the
  embedding ensemble's 0.9975. The same rule over a handcrafted 135-dim action-statistics
  view achieves 0.998. The calibration therefore demonstrates that the pipeline is wired
  correctly and that a ranked worklist is non-vacuous; it licenses nothing about the
  embedding auditor, the sentence encoder, or the ensemble as a mechanism.
- **"The detector finds label errors." REFUTED for the error type that matters.** Under
  harder perturbations the ensemble degrades to AUC 0.955 (swap to the visually nearest task)
  and 0.913 (swap to the semantically nearest task, recall@200 0.78). Under **correlated**
  corruption — an entire task relabelled with one wrong instruction, which is exactly what a
  real annotation pipeline produces from one bad annotator or one bad template — it is at
  **pure chance: AUC 0.487–0.511, precision@50 0.03–0.19, recall@200 0.10–0.14.** The
  mechanism is structural, not incidental: when an episode's neighbours carry the *same*
  wrong label, neighbourhood disagreement is zero by construction. **The method finds
  isolated label errors and is blind to systematic ones.**
- **"The ensemble helps." Survives only in the weakest form.** Paired across the same 10
  seeds the gain is real but tiny: +0.00095 AUC (p=0.003, 10/10 wins) at S=85, +0.00098 at
  S=169. At S=85 the gain on P@50, P@200 and recall@200 is *exactly zero* in every seed
  because all are at ceiling. A 0.001 AUC improvement is not an operational argument for
  running two encoders.

## Manual inspection of the top 50 (orchestrator, contact sheets)

Classification requested by the roadmap — wrong label / genuinely rare behaviour / detector
artifact: **0 wrong labels, 0 rare behaviours, ~50 detector artifacts.** That is the expected
answer for a corpus with no label noise, and it is why the swap calibration rather than this
list was the load-bearing measurement.

Every suspect inspected belongs to a **visually ambiguous task family** — a shared scene in
which the instruction differs only by spatial referent or goal. Rank 1 (ep 1327, "pick up the
black bowl *between the plate and the ramekin*") sits in a scene with three near-identical
black bowls, where tasks 34/36/37/38/30 differ only in which bowl the sentence picks out.
Rank 2 ("put the bowl on top of the cabinet") shares its kitchen scene with task 17, "put the
bowl on the stove"; the two are indistinguishable until the destination diverges at the end.
Rank 4 ("pick up the chocolate pudding") is the grocery-basket scene where many tasks share
one object set. The task histogram confirms this is not cherry-picking: 13 of the top 50 are
task 34 alone, and ~35 of 50 come from these shared-scene families. The control episode drawn
for comparison was task 37 — the *same* family as rank 1 — and scored exactly zero, which
shows the score is driven by whether an episode's 10 nearest visual neighbours happen to land
on its own task, not by anything wrong with its label. The verifier's independent finding
matches: **100% of wrong-task neighbours are within-suite** (140/140 DINOv2, 142/142 CLIP).

## A finding about rank-averaging in a degenerate score regime

The Gate A decision was to build ensemble suspect lists by **rank-averaging** across encoders
rather than intersecting top-K sets. That is sound where scores are continuous, as in Stage A.
Here it misbehaves, and the reason is worth recording: 1,595 of 1,693 episodes score *exactly*
zero, so rank-averaging admits an episode on one encoder's evidence while the other contributes
a tied mid-rank of ~877 that is indistinguishable from "no opinion". Concretely, of the top 50:
**23 episodes have both encoders nonzero; 27 have one encoder at exactly 0.0**; and of the
top 200, **102 rows have both encoders at zero and are pure tie-break noise.** Rank-averaging
behaves like a union rather than a consensus when most of the mass is tied.

The honest operative output for this corpus is **the 23 consensus episodes**, not a top-200
list. Recommendation, scoped narrowly: keep rank-averaging as the default, but use
intersection — or require both encoders nonzero — whenever the score distribution is
degenerate. This is not a reversal of the Gate A decision; it is a boundary condition on it.

## Cross-encoder tail instability, worse than Stage A

Stage A's standalone methodological finding reproduces here and intensifies. Across the two
*vision* encoders: Spearman 0.361 over all episodes, top-50 overlap 0.34, top-200 overlap 0.20
— against Stage A's sentence-encoder pairs at 0.88–0.94 global and 47–57% top-200. The Gate A
premise-checker predicted exactly this ("cross-modal spaces will almost certainly be less
stable than the four same-modality encoders tested"), and it was right. Note the caveat that
much of this tail is the degenerate zero block; the comparison is still directionally sound
because the same snapping and independent-seed protocol was applied throughout.

## Could not run / could not verify

- **LIBERO's own annotator agreement α, as a comparator for DROID's 0.8125** — impossible.
  One instruction per task, one task per episode, no annotator fields. No proxy substituted.
- **`lerobot/droid_100` real-data spot-check** — not run, recorded as untested. Its 461 MB of
  video would have taken ~61 min against a 30-min budget, but the decisive reason is content:
  0 of 100 episodes carry more than one annotation and 54 of 100 have an empty-string
  instruction. Side finding, deliberately **not** generalised: a widely-used LeRobot conversion
  of DROID drops the multi-annotator fields and blanks 54% of instructions. Whether the full
  `lerobot/droid` conversion shares this is **unchecked** and must not be asserted.
- **Whether any of this transfers to DROID** — untested by construction. See below.

## Premise check (Gate B) — INVALIDATED. Hard stop.

Fresh premise-checker, minimal context. Premise under test: *"Embedding-distance methods can
audit whether a label describes what the camera saw; ranking episodes by cross-modal
neighbourhood disagreement identifies mislabeled episodes, and validating this on LIBERO
justifies proceeding to Stage C."*

**Verdict: INVALIDATED.** Deciding number: **AUC 0.487–0.511 under correlated corruption**,
exact chance, with precision@50 of 0.03–0.19. Correlated corruption is the failure mode real
annotation pipelines produce; against it the ranked list identifies nothing. Two supporting
facts made this a kill rather than a dent: the detector produces no ranking at all over 94% of
the corpus (1,595 of 1,693 exactly tied), and the trivial `task_index` baseline strictly beats
it at AUC 1.0000 without using the method under test.

What survives is real but is not the claim: visual embeddings encode trajectory-level task
information beyond static scene (LIBERO-Goal frame-0 0.108 → full-episode 1.000), and
confusions land on semantically near instructions (0.771 vs 0.444).

**Was LIBERO a valid validation corpus? No — structurally, not incidentally.** Its language
view is 40 points each duplicated ~42 times, in bijection with `task_index` and near-bijection
with scene identity, so "language neighbourhood" reduces exactly to "`task_index` equality
class" and the cross-modal apparatus collapses into a lookup. LIBERO **cannot discriminate**
the hypothesis under test from the null "the dataset has 40 blocks"; the thumbnail result is
the empirical confirmation. It is confounded on a second axis too: with no true errors to
find, precision is assessable only against injections, and the natural injection is precisely
the trivially solvable case. **Zero wrong labels in the top-50 inspection is therefore
uninformative about precision — recall of an empty set is undefined.**

**What transfers to DROID:** (1) the correlated-corruption mechanism, as a property of the
*method family* rather than of LIBERO — neighbourhood-disagreement detectors are blind to
noise that does not break local visual–language correspondence, and when a corrupted group
constitutes its own visual neighbourhood the score is zero by construction; on DROID this
bites wherever mislabeling is batched. (2) Cross-encoder tail instability (Spearman 0.361,
top-200 overlap 0.20): a "top-N suspicious episodes" deliverable is not well-defined.
(3) The methodological warning that visual k-NN task purity is ~two-thirds static scene
recognition and is matched by 8×8 thumbnails — any paper reporting such purity without
frame-0 and thumbnail baselines is reporting scene ID.

**What does not transfer: every positive number in this stage.** AUC 0.996–0.998,
precision@50 = 1.000, and purity 0.9976 must not be carried into a DROID claim in any form.

## Surprises

- The strongest attack on the headline came from a **192-number 8×8 thumbnail**. Nothing in
  the roadmap anticipated that the foundation-model encoders would be unnecessary.
- The detector's blindness to correlated corruption (AUC 0.49) is the most consequential
  result of the stage, and it was not on the original measurement list at all — it exists only
  because a verifier constructed a harder perturbation than the one that was briefed.
- LIBERO's structure — 40 instructions in bijection with 40 `task_index` classes — is what
  makes the trivial baseline available and the language view degenerate. DROID has 64,516
  unique strings among 125,276 annotations and no `task_index`, so neither pathology exists
  there. The corpus chosen to validate the method is structurally the least suitable one for
  demonstrating that the method does anything a lookup could not.

## Open questions for the owner — surfaced, not acted on

These are research-judgement calls and a hard stop is in force. Recorded so they are not lost.

1. **A converging negative result may be the more valuable contribution.** Stage A found
   single-encoder per-episode rankings ~50% unstable at the tail; Stage B found visual
   cross-encoder top-200 overlap of 0.20. The project has now independently failed to produce
   a stable per-episode embedding ranking in *both* modalities. The defensible synthesis —
   *aggregate embedding statistics over a corpus are robust; per-episode embedding rankings
   are encoder artifacts at the tail* — is supported by both stages and warns against a
   common informal practice. That framing makes Stage B a component of the paper rather than
   a hole in it.
2. **Stage C as written has lost both halves of its deliverable.** "Convert a measured noise
   rate into a predicted performance cost" needs a measured noise rate, which Stage B was
   supposed to supply and cannot. Worse, a scalar rate is now known to be the wrong
   parameter: i.i.d. uniform label noise is the *easiest* kind for a policy to absorb, so a
   curve fitted on it would under-estimate the cost of real noise — reproducing Stage B's
   error in a more expensive medium. If a Stage C runs it needs a rate × correlation-structure
   family: i.i.d. swap (optimistic bound), semantically-near swap, block relabeling
   (pessimistic bound), and paraphrase-only perturbation.
3. **The paraphrase-cost experiment is the one that follows from a verified result.** It
   needs no working detector and answers a question the project already owns evidence for:
   does a policy pay for seeing 64,516 unique strings describing a much smaller behaviour set?
4. **The cheap missing prerequisite: a human audit of ~200 random DROID episodes against their
   instructions**, to establish a base rate. That is the missing x-value for any cost curve and
   the only honest denominator for evaluating any detector. It costs a fraction of Stage B and
   arguably should have preceded it.
5. **The α = 0.81 headline itself argues against the audit framing.** If annotators agree
   semantically and rarely disagree in meaning, DROID's true mislabeling base rate is plausibly
   low, and chasing a low base rate demands precision this method family has not shown. The
   pathology the data exhibits is label *variance*, not label *error* — which relocates the
   cost from the audit side to the training side.
6. **Do not cite the 54%-empty-instruction figure yet.** It is from a 100-episode subset; Stage
   A's 125,276 annotations over 64,516 strings suggest the annotated portion is well populated,
   so it is plausibly a shard or format artifact. Cheap to check on the full corpus; check
   before citing.
