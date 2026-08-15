# Working in this repo

Context for any agent session operating on `vla-label-audit`.

## What this is

A research project measuring whether the natural-language labels in robot-learning
datasets actually describe the trajectories they are attached to.

The claim being built toward: vision-language-action models are known to largely ignore the
language they are given — [RoboSemanticBench](https://arxiv.org/html/2606.02277) reports
80–100% grasp success but only 2–21% *commanded-task* success. The field reads that as an
architecture problem. Nobody has checked whether the training labels were correct. This
project checks.

Owner: Adrianos Botsios (Brown, CS + Econ). This is real research intended for an arXiv
preprint and a workshop submission, not a toy.

## Where it's going

| Session | Work | Status |
|---|---|---|
| 1 | DROID annotator agreement — text only, 12 MB JSON, no video | in progress |
| 2 | LIBERO images, build the joint vector index, find wrong labels | not started |
| 3 | Inject known label noise, train tiny policies, fit the cost curve | not started |
| 4 | Write-up | not started |

## Hard rules

**Never weaken a test to make it pass.** The tests in this repo are correctness claims, not
smoke checks. `test_semantic_matches_the_naive_implementation` pins the fast path to the slow
one at 1e-10 — if it fails, the fast path is wrong, not the tolerance.
`test_bootstrap_memory_does_not_scale_with_episode_count` guards a bug that allocated 59 GB.
`test_mde_actually_delivers_the_nominal_power` validates an analytic formula against
simulation. Loosening any of these silently destroys the point of the project. If a test
fails, fix the code or stop and ask.

**Never fabricate a number, a result, or a citation.** If a run did not complete, say it did
not complete. A made-up plausible figure in a statistics project is worse than no figure,
because nobody can tell by looking.

**If you change a statistical formula, prove it.** Add a test comparing the new
implementation against a naive/brute-force version on small inputs, or against an analytic
result where one exists. That is how every function in `stats`-adjacent modules here was
validated.

**Do not install into system Python.** Use the `.venv` in this directory. The system Python
3.11 on this machine already had numpy force-upgraded once, which broke a TensorFlow install.

**Run `pytest -q` before claiming anything works.** Expect 50 passing.

**Ask rather than guess on research judgement.** "Is this metric right", "is this finding
novel", "does this number support the claim" are not coding questions. Surface them; don't
resolve them silently.

## Design decisions and why — do not casually undo these

**Exact nearest neighbours, not approximate.** `knn_indices` and the planned FAISS index use
exact search (`IndexFlatIP`). At this scale — under a million vectors at 384 dims — brute
force is minutes and under a gigabyte. Exactness removes a whole class of "did ANN recall
cause that result?" objections before a reviewer raises them. Do not swap in HNSW/IVF for
speed we do not need.

**`scalable.py` exists for a specific reason.** The textbook Krippendorff's alpha needs an
N×N difference matrix; at DROID scale (~150k annotations) that is 90 GB. The module computes
the identical quantity in O(n·d²) using a closed form for expected disagreement:
`sum_ij (1-s_ij)^2 = n^2 - 2||sum_i x_i||^2 + ||X^T X||_F^2`. Exact, not approximate. The
naive path in `agreement.py` is kept deliberately, as the reference the fast path is tested
against. Do not delete it.

**Fidelity, not accuracy, where the two compete.** Offline loss against demonstrations is a
documented poor predictor of closed-loop robot success. Metrics that compare a modified model
to *the unmodified model's own output* sidestep that; metrics against ground-truth
demonstrations are reported only as an explicitly-labelled weak proxy.

**Episode-clustered bootstrap, never i.i.d.** Frames and annotations within an episode are
correlated. Resampling individual rows understates the spread, sometimes by 3× or more —
there is a test demonstrating exactly this. Always resample whole episodes.

**Semantic *and* nominal agreement are both reported.** The gap between them is the paraphrase
rate, which is a finding in its own right. Do not drop one for brevity.

**`svd_project`-style spectral tools are diagnostics, not performance ceilings.** Eckart–Young
bounds reconstruction of a matrix, not downstream task fidelity. Do not restate it as a bound
on model performance.

## Machine and environment facts

- Apple M2 Pro, 12 cores, **16 GB unified memory**, macOS 26.5, ~530 GB free.
- MPS works. **`is_amp_available("mps")` is `False`** — training is fp32, no autocast.
- **M2 has no AV1 hardware decode** (Apple added it in M3). LIBERO is AV1. Decode once, cache
  embeddings to disk immediately, never decode twice. **Measured 2026-08-15:** software decode
  via PyAV/libdav1d runs at ~2,300–2,500 img/s sequentially at 256×256, so the full 546,930-image
  corpus is ~4 CPU-minutes — the lack of hardware decode turned out not to bite. But *sequential
  whole-file* decode is ~200× cheaper per frame than per-frame keyframe seeking, so iterate video
  files once in order; never seek per frame.
- Python 3.11 in `.venv`. **LeRobot requires ≥3.12** — that install is a session-2 prerequisite.
- `faiss-cpu` has an arm64 wheel and works; `hnswlib` is sdist-only and will try to compile.
- There is a known historical MPS bug where `.to(device, non_blocking=True)` returned garbage
  ([lerobot#496](https://github.com/huggingface/lerobot/issues/496)). Sanity-check any MPS
  result against CPU before trusting it — that class of bug fails silently.

## Choices already made for session 2 — do not re-litigate without asking

- **Dataset: `lerobot/libero`** (1.94 GB, 1,693 episodes, 273,465 frames, 546,930 images, 40
  instructions). NOT `HuggingFaceVLA/libero` (69.86 GB) or `physical-intelligence/libero`
  (34.94 GB). **Correction, 2026-08-15:** the note that this variant is "PNG-in-parquet instead
  of AV1 video" was **wrong**. Downloaded and verified: it is LeRobot **v3.0** with images stored
  as **AV1 MP4** (37 files per camera, many episodes concatenated per file); the parquet files
  hold only state/action/index columns. The dataset choice itself stands — all counts match
  exactly and it is the smallest variant — only the stated reason was incorrect. Decode cost is
  negligible in practice (see the AV1 note above).
- **Encoder: DINOv2 ViT-S/14** (384-d), not ViT-B/14 (768-d). Halves both embedding time and
  index memory, which matters on 16 GB of *unified* memory shared with the GPU.
- **Never ACT, Diffusion Policy, or SmolVLA fine-tuning.** Reported at **13+ hours per run** on
  Apple Silicon; a seeded comparison would take a week. Session 3 uses a state-based MLP
  behaviour-cloning policy on PushT — roughly 2 minutes per run — which is what makes the
  experiment possible at all. This is the single most important engineering constraint in the
  project.
- **PushT install trap:** pin `pymunk<7.0.0`, or gym-pusht fails with
  `'Space' object has no attribute 'add_collision_handler'`.

## Style

Match what is there. Numpy-style docstrings that explain *why* a thing is done, not just what
it does. Tests get docstrings when the reason for the test is not obvious from its name.
Prefer clarity over cleverness; this code is meant to be read by reviewers.
