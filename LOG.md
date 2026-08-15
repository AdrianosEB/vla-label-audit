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
