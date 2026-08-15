# Log

One paragraph per event, newest last. Written by the orchestrator so a human can catch up
asynchronously without reading everything.

---

**2026-08-15 — Stage A opened (encoder robustness).** State check: `pytest -q` 50 passed;
full-corpus MiniLM embeddings cached. Plan: executor writes `scripts/encoder_robustness.py`
(embed/analyze modes, per-encoder caches); three executors embed the 125,276-sentence corpus
with mpnet, gte-base, bge-large in parallel; one executor runs the full analysis; 3 adversarial
verifiers with distinct lenses; then premise check at Gate A. Encoder set: MiniLM-L6-v2
(baseline), all-mpnet-base-v2, thenlper/gte-base, BAAI/bge-large-en-v1.5, TF-IDF floor.
gte-base chosen over e5 deliberately — e5's query/passage prefix convention is a methodological
trap for symmetric similarity. Nothing in `vla_label_audit/` is to be touched.

**2026-08-15 — Stage A pipeline validated.** `scripts/encoder_robustness.py` written by executor,
smoke-tested end-to-end on the 2,000-episode slice (`--boot 30`), pytest 50 passed before and
after. Smoke alphas: minilm +0.827, mpnet +0.837, gte +0.825, bge-l +0.840, tfidf +0.658; lab-rank
Spearman 0.90–1.00, episode-rank 0.75–0.93. Two observations carried forward: (1) gte embeds to
float16 (model ships fp16 weights) — quantization ~1e-3, to be verified as harmless; (2) gte's
D_e is 0.039 vs MiniLM's 0.465 — a 10x more anisotropic space that still lands alpha in the same
band; the "does alpha track D_e" check is now a mandatory verifier lens. Full-corpus caches:
minilm (pre-existing), mpnet (125276x768 fp32), gte (125276x768 fp16) done; bge-l running.

**2026-08-15 — Stage A full-corpus analysis complete; verification in flight.** Neural alphas:
minilm +0.8125 [+0.8107,+0.8140], mpnet +0.8218, gte +0.8095, bge-l +0.8165 — band width 0.012.
TF-IDF floor +0.6310 (vocab 2,442, 8 zero-vector rows dropped). D_e spans 0.040 (gte) to 0.475
(minilm) — 12x — while alpha moves 0.012, so alpha does not visibly track anisotropy. Per-episode
rank Spearman 0.88–0.94 (neural pairs); per-lab 0.79–0.99; CLVR lowest and GuptaLab highest in
all five arms, middle of the ordering shuffles. MiniLM arm reproduced the stored baseline
bit-identically; ROADMAP's recorded CI lower bound "0.8109" is a transcription typo for 0.8107
(both JSONs agree) — to be corrected in the state update, not a measurement contradiction.
Three adversarial verifiers dispatched: statistical validity, confounds (identical-string ties,
length, lab-size, fp16), independent reproduction (incl. fast-vs-naive path and from-scratch
nominal alpha). Gate A report after their verdicts.

**2026-08-15 — Stage A closed at Gate A: premise WEAKENED. Halted for human review.** The
previous session crashed mid-verification; verifiers were re-dispatched fresh. Reproduction
verifier: everything matched — pytest 50/50, fast path = naive path to machine precision,
from-scratch nominal alpha exact to the last digit, full-corpus alphas to ~1e-14, caches
verified by content hash and re-embedding. Survived all three lenses: the cross-encoder band
(0.8095–0.8218, width 0.012, not manufactured by string ties), the TF-IDF floor (0.631, ≥0.14
below every lexical variant), alpha's independence from anisotropy (D_e 20x range), and the
paraphrase gap (0.756–0.768). **Refuted as originally worded:** (1) the ranked worst-episode
list — top-200 overlap between encoders is only 47–57% (global Spearman 0.88–0.94 is bulk-
dominated); survives only as a neighborhood/pool claim (93–99% of top-200 within another's
top-2000). (2) CLVR-lowest — P(CLVR=argmin)=0.73 under MiniLM, flips vs RAIL under tie
removal; GuptaLab-highest fully robust. Premise checker (fresh, minimal context): WEAKENED —
the paraphrase-gap thesis holds, but Stage B's plan to rank episodes by cross-modal embedding
disagreement sits exactly at the granularity that proved unstable; its brief now requires
ensemble/consensus suspect selection across ≥2 vision encoders. ROADMAP state updated (incl.
correcting the 0.8109→0.8107 CI transcription typo); full report in `results/stage-A.md`.
Stopped at Gate A per instruction — Stage B not started.

**2026-08-15 — Gate A approved by owner; Stage B opened under revised brief.** Owner decisions:
(1) ensemble suspect selection by **rank-averaging** across encoders, not top-K intersection;
(2) lab claim worded as "GuptaLab uniquely on top, {CLVR, RAIL} the low cluster"; (3) the
tail-instability caveat applies retroactively to Session 1's MiniLM-only `worst_episodes`
list in `data/droid_agreement_results.json` — treat as pool sample, recompute any released
list by rank-average over the four cached encoders; (4) tail instability recorded as a
standalone methodological finding in `results/stage-A.md`. Stage B plan per ROADMAP + revised
brief: acquire `lerobot/libero` (1.94 GB), embed 546,930 images with DINOv2 ViT-S/14 plus a
second vision encoder for the ensemble, build the joint exact-search index, run the
crossmodal battery, rank suspects by cross-encoder rank-average, inspect top ~50, verify
adversarially, premise-check, stop at Gate B.

**2026-08-15 — LIBERO acquired; two structural facts force a Stage B redesign.** Download
verified: 1,693 episodes / 273,465 frames / 2 cameras / 546,930 images / 40 instructions — every
count matches the recorded expectation exactly. Two corrections and two consequences. (a) The
docs recorded `lerobot/libero` as "PNG-in-parquet instead of AV1 video"; that is **wrong** — it is
LeRobot v3.0 with AV1 MP4 video, parquet holding only state/action/index. The dataset choice
stands (smallest variant, counts exact); only the stated reason was wrong, and it is corrected in
`CLAUDE.md`. Decode turned out cheap anyway: ~2,400 img/s sequential via libdav1d, ~4 CPU-min for
the whole corpus, though sequential whole-file decode is ~200x cheaper than per-frame seeking.
(b) **LIBERO has no multiple annotations per episode** — the planned DROID comparator α is
impossible, not merely awkward; reported as impossible, no proxy. (c) **Only 40 distinct
instructions across 1,693 episodes** (29/44/50 per instruction) — the language view has ≤40 unique
vectors, so the planned language-vs-visual k-NN Jaccard `neighborhood_overlap` is largely a
tie-breaking artifact. Headline measurement reframed to **visual-neighbourhood task purity**
against a ~2.6% chance baseline, with the degenerate overlap still reported so the reason is on
record. (d) Because LIBERO's instructions are the task definitions that generated scripted
simulated demos, its natural label-noise rate is ≈0 by construction — an audit there measures
detector *specificity*, not detection ability. Added a **synthetic label-swap validation** (swap
instructions across task suites, report ROC/AUC and precision@50) to measure recall and calibrate
the method; this is a deliberate addition to the brief and is flagged at Gate B. Embedding
executor dispatched: DINOv2 ViT-S/14 (primary, self-supervised) + CLIP ViT-B/32 (ensemble partner,
language-supervised — noted as partly circular for judging image-label match, so DINOv2 leads).

**2026-08-15 — Label-swap calibration approved; three owner additions.** (1) The
synthetic-in-simulation caveat is to be stated plainly as the **top** limitation in
`results/stage-B.md` and any writeup — LIBERO has no real annotation noise to find, planted
cross-suite swaps are easier than plausible-but-wrong human labels, so any AUC/precision reported
is an upper bound on real-world performance and must be described as one. (2) Report
**precision@50 and precision@200 separately** — Gate A showed the tail is where ranking
instability lives, so a lone top-50 figure would flatter the method. (3) Time-boxed probe of
`lerobot/droid_100` (~30 min hard budget) dispatched as a qualitative real-human-annotation
sanity check, not a ranking study; if it exceeds budget it is skipped and recorded as untested.
The probe also checks two things worth knowing regardless: whether the LeRobot conversion
preserved DROID's `language_instruction_2`/`_3` multi-annotator fields, and whether its episodes
can be joined to our 50,092-episode annotation corpus by episode key.

**2026-08-15 — droid_100 spot-check NOT RUN; recorded as untested. Failed on content, not cost.**
Metadata probe was cheap (2.85 MB, 6.8 s; LeRobot v3.0, 100 episodes, 32,212 frames, 3 cameras,
fps 15, 180x320 AV1 — note the hub README is stale v2.0 text and should not be trusted). The
461 MB of video would have taken ~61 min at a measured 123–132 KB/s, 2x the budget, so it was
aborted — though that rate is unauthenticated on a link contended by an unrelated training job,
and an HF_TOKEN might change it. The decisive reason is content, not cost: **0 of 100 episodes
have more than one annotation** (no `language_instruction_2`/`_3` field survived the conversion),
and **54 of 100 have an empty-string instruction** — only 46 carry text. No episode identifier is
exposed, so a join back to our corpus is text-only: 26 unique-by-text (unverifiable), 20
ambiguous, 54 unjoinable. droid_100 therefore adds no annotation signal we lack, and its video
would buy at most 26 text-joined episodes. **Side finding, deliberately not generalised:** a
widely-used LeRobot conversion of DROID drops the multi-annotator fields and blanks 54% of
instructions — a label-quality defect in the tooling layer, squarely on this project's topic.
Whether the full `lerobot/droid` conversion shares it is UNCHECKED and must not be asserted;
verifying it costs one cheap metadata pull and is logged as an open follow-up for the owner.

**2026-08-15 — Stage B ran, was verified, and its premise was INVALIDATED at Gate B. HARD STOP;
Stage C not started; awaiting a human.** Embedded all 546,930 LIBERO images with DINOv2 ViT-S/14
and CLIP ViT-B/32 (~87 min on MPS; MPS-vs-CPU agreement to fp32 noise after pinning preprocessing
to CPU, since the MPS bicubic resize disagrees with the CPU kernel by ~0.35 and would have made
the check measure the resampler; alignment verified with ±1 and +50-row shift controls). Ran the
reframed battery, then three adversarial verifiers. **All arithmetic reproduced independently**
— views rebuild bit-exactly, purity/chance/nonzero-counts/tail-stats/swap-metrics all match, CCA
held-out path confirmed leak-free (permutation control 0.9925→0.025). **But two of the three
substantive claims were refuted, and the premise was killed.** (1) Task purity 0.9976 is computed
from `task_index` and is *bit-identically* invariant to permuting task ids or replacing all 40
instructions with nonsense — it never reads the language; an 8×8 RGB thumbnail (192 numbers, no
network) reaches 0.9988, matching DINOv2; "40× chance" is the arithmetic ceiling. (2) The swap
calibration is correct arithmetic but proves nothing about embeddings — a trivial `task_index`
mismatch rule using no language embedding and no cosine distance gets AUC 1.0000, beating the
ensemble's 0.9975 — and under **correlated corruption (a whole task relabelled) the detector is
at pure chance, AUC 0.487–0.511, P@50 0.03–0.19**. That last number decided the gate: it is the
failure mode real annotation pipelines produce, and the method is structurally blind to it
because when neighbours share the wrong label, disagreement is zero by construction. Also: the
detector is degenerate on unmodified LIBERO (98/1,693 nonzero, 1,595 exactly tied, 23 flagged by
both encoders); manual inspection of the top 50 contact sheets found **0 wrong labels, 0 rare
behaviours, ~50 detector artifacts** from visually ambiguous task families — uninformative about
precision, since recall of an empty set is undefined; cross-encoder tail agreement is worse than
Stage A (Spearman 0.361, top-200 overlap 0.20). Premise-checker: LIBERO was **structurally** the
wrong validation corpus — its 40 instructions are in bijection with `task_index`, so the
cross-modal apparatus collapses into a lookup and cannot discriminate the hypothesis from the
null "the dataset has 40 blocks". What survived: a real trajectory-dependent signal (LIBERO-Goal
frame-0 0.108 → full 1.000) and confusions landing on semantically near instructions (0.771 vs
0.444). **No positive Stage B number transfers to DROID.** Stage A's headline is untouched as a
measurement. Owner-facing options (converging negative result as a contribution; DROID base-rate
human audit; paraphrase-cost experiment; rate × correlation noise family for any Stage C) are
recorded at the end of `results/stage-B.md` as questions, not decisions.
