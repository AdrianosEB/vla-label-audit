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
