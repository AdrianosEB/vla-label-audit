"""Does the DROID agreement number survive a change of sentence encoder?

Every semantic-alpha figure this project reports is conditional on one model:
all-MiniLM-L6-v2 decides what "means the same thing". A reviewer's obvious
objection is that the headline is a fact about MiniLM's geometry, not about the
annotators. This script answers it by recomputing the full analysis under four
independent encoders spanning size (384-d to 1024-d), training recipe, and
vendor, plus a deliberately non-neural TF-IDF floor that knows nothing about
meaning beyond shared vocabulary.

Two different robustness claims are checked, because they can fail separately:

* **Level**: does alpha itself move? Encoders squash cosine similarity into
  different ranges, so some drift in the absolute number is expected and is not
  by itself damning -- which is why D_o and D_e are reported alongside alpha.
* **Ordering**: do the encoders agree on *which labs* and *which episodes* are
  the bad ones? The audit's operational output is a ranked worklist; if the
  ranking is encoder-specific, the worklist is an artefact. Spearman rank
  correlations over per-lab alphas and per-episode disagreement scores test
  exactly this, pairwise across all arms.

Embedding 125k sentences with the larger models takes tens of minutes each, so
embedding and analysis are separate passes: run `--embed-only` once per encoder
(resumable, cached), then `--analyze` reads the caches and never embeds.

Run:
    python scripts/encoder_robustness.py --embed-only --encoder mpnet
    ...one per encoder, then:
    python scripts/encoder_robustness.py --analyze
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from droid_agreement import CACHE, corpus_tag, embed, fetch_annotations, flatten  # noqa: E402

from vla_label_audit.scalable import (  # noqa: E402
    alpha_nominal,
    alpha_semantic,
    bootstrap_alpha_semantic,
    per_unit_disagreement,
)

# Chosen to differ in the ways that could plausibly matter: mpnet is the same
# vendor but a larger backbone, gte is a different lab and training recipe,
# bge-l is a different lab *and* 1024-d. If alpha survives all three, "it's a
# MiniLM artefact" is off the table.
NEURAL = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "gte": "thenlper/gte-base",
    "bge-l": "BAAI/bge-large-en-v1.5",
}
ENCODER_ORDER = [*NEURAL, "tfidf"]

TFIDF_MAX_FEATURES = 5_000
# One bootstrap replicate costs O(n d^2) in the Gram matmul; at d ~ 2.5-5k the
# TF-IDF arm is 40-170x the per-replicate cost of MiniLM. 50 replicates keeps
# the CI honest-but-wide instead of making this arm the wall-clock bottleneck.
TFIDF_N_BOOT = 50
MIN_LAB_EPISODES = 30


def encoder_tag(model: str, texts: list[str]) -> str:
    """`droid_agreement.corpus_tag` with the model name as a parameter.

    The original hard-codes MiniLM into the hash, so every other encoder would
    collide onto the same cache file and silently reload MiniLM vectors. Same
    construction, same failure-proofing (content-hashed, not length-keyed),
    just parameterised. For the MiniLM model name this reproduces
    ``corpus_tag`` exactly, which is what lets `--analyze` locate the existing
    baseline cache without re-embedding.
    """
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return f"{len(texts)}_{h.hexdigest()[:12]}"


def encoder_cache_path(key: str, texts: list[str]) -> Path:
    """Where a given encoder's vectors for this exact corpus live on disk.

    MiniLM keeps the legacy un-slugged filename so the embeddings computed by
    `droid_agreement.py` are reused rather than duplicated; every other
    encoder gets its slug in the name so a directory listing is self-describing.
    """
    if key == "minilm":
        return CACHE / f"embeddings_{corpus_tag(texts)}.npy"
    return CACHE / f"embeddings_{key}_{encoder_tag(NEURAL[key], texts)}.npy"


def embed_neural(key: str, texts: list[str]) -> np.ndarray:
    """Embed the corpus with one encoder, cached to disk keyed by content.

    batch_size is 128 rather than droid_agreement's 256 because bge-large at
    1024-d is a 1.3 GB model sharing 16 GB of unified memory with its own
    activations; 256-sentence batches OOM the MPS allocator there.
    """
    if key == "minilm":
        # Delegate entirely so the existing full-corpus cache is hit and any
        # future change to the baseline path happens in exactly one place.
        return embed(texts, corpus_tag(texts))

    CACHE.mkdir(exist_ok=True)
    path = encoder_cache_path(key, texts)
    if path.exists():
        print(f"  using cached embeddings at {path}")
        return np.load(path)

    from sentence_transformers import SentenceTransformer
    import torch

    model_name = NEURAL[key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  embedding {len(texts):,} sentences with {model_name} on {device}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    np.save(path, emb)
    return emb


def tfidf_arm(texts: list[str]) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Dense TF-IDF matrix plus a mask of rows that survived.

    This arm exists as a floor: it has no notion of meaning beyond shared
    vocabulary, so if the neural encoders only barely beat it, the "semantic"
    in semantic alpha is doing very little work. Dense because the alpha
    closed form is a Gram-matrix computation; at ~2.5k vocabulary that is
    low single-digit GB, which is fine.

    A handful of annotations are pure punctuation ("+++++") and vectorise to
    exactly zero, on which cosine distance is undefined -- `alpha_semantic`
    refuses them by raising. For this arm only those rows are dropped and
    counted; the neural arms embed every string, so they keep the full corpus.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True)
    mat = vec.fit_transform(texts)
    vocab = len(vec.vocabulary_)
    capped = vocab > TFIDF_MAX_FEATURES
    if capped:
        vec = TfidfVectorizer(lowercase=True, max_features=TFIDF_MAX_FEATURES)
        mat = vec.fit_transform(texts)
    dense = np.asarray(mat.todense(), dtype=np.float32)
    keep = np.linalg.norm(dense, axis=1) > 0
    return dense[keep], keep, vocab, capped


def analyze_arm(
    episodes: np.ndarray, emb: np.ndarray, n_boot: int
) -> tuple[dict, dict[str, float]]:
    """Corpus alpha, per-lab alphas, and per-episode scores for one encoder.

    Per-lab filtering is on *multiply-annotated* episodes (``n_units``), not
    raw episode count: an episode with a single annotation contributes nothing
    to alpha, so a lab of 100 singletons has no agreement to estimate.
    """
    sem = alpha_semantic(episodes, emb)
    _, lo, hi = bootstrap_alpha_semantic(episodes, emb, n_boot=n_boot, seed=0)

    labs = np.array([e.split("+")[0] for e in episodes])
    per_lab: dict = {}
    for lab in np.unique(labs):
        m = labs == lab
        try:
            res = alpha_semantic(episodes[m], emb[m])
        except ValueError:  # no multiply-annotated episode at all
            continue
        if res.n_units < MIN_LAB_EPISODES:
            continue
        _, lab_lo, lab_hi = bootstrap_alpha_semantic(episodes[m], emb[m], n_boot=n_boot, seed=0)
        per_lab[str(lab)] = {
            "alpha": res.alpha,
            "ci": [lab_lo, lab_hi],
            "n_episodes": int(res.n_units),
        }

    scores = {str(k): v for k, v in per_unit_disagreement(episodes, emb).items()}
    summary = {
        "dims": int(emb.shape[1]),
        "alpha": sem.alpha,
        "ci": [lo, hi],
        "d_o": sem.observed,
        "d_e": sem.expected,
        "n_boot": n_boot,
        "per_lab": per_lab,
    }
    return summary, scores


def load_corpus(limit: int | None) -> tuple[np.ndarray, list[str]]:
    raw = fetch_annotations()
    print(f"loaded {len(raw):,} episodes from the annotation file")
    episodes, texts, _ = flatten(raw, limit)
    print(f"  {len(texts):,} annotations across {len(set(episodes.tolist())):,} episodes")
    return episodes, texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--embed-only", action="store_true", help="compute+cache one encoder's embeddings, then exit"
    )
    mode.add_argument("--analyze", action="store_true", help="run the analysis over cached embeddings")
    ap.add_argument("--encoder", choices=sorted(NEURAL), help="which encoder (with --embed-only)")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N episodes")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap replicates (neural arms)")
    args = ap.parse_args()

    if args.embed_only:
        if not args.encoder:
            ap.error("--embed-only requires --encoder")
        _, texts = load_corpus(args.limit)
        t0 = time.perf_counter()
        emb = embed_neural(args.encoder, texts)
        print(
            f"  {args.encoder}: {emb.shape[0]:,} x {emb.shape[1]} embeddings ready"
            f" in {time.perf_counter() - t0:.1f}s"
        )
        return

    episodes, texts = load_corpus(args.limit)

    # Analysis must never silently fall into an hour of embedding: missing
    # caches are a usage error ("run --embed-only first"), not work to do.
    missing = [k for k in NEURAL if not encoder_cache_path(k, texts).exists()]
    if missing:
        raise SystemExit(
            "missing cached embeddings for: " + ", ".join(missing) + "\n"
            "run, for each (with the same --limit as this analysis):\n"
            + "\n".join(
                f"  python scripts/encoder_robustness.py --embed-only --encoder {k}"
                for k in missing
            )
        )

    # Nominal alpha compares strings, so it is the one encoder-independent
    # quantity here; computed once and shared by every arm's paraphrase gap.
    nom = alpha_nominal(episodes, texts)
    print(f"\nnominal alpha (encoder-independent): {nom.alpha:+.4f}")

    results: dict[str, dict] = {}
    episode_scores: dict[str, dict[str, float]] = {}

    for key in NEURAL:
        print(f"\n[{key}] {NEURAL[key]}")
        emb = embed_neural(key, texts)  # cache hit guaranteed by the check above
        t0 = time.perf_counter()
        summary, scores = analyze_arm(episodes, emb, args.boot)
        summary["gap"] = summary["alpha"] - nom.alpha
        results[key] = summary
        episode_scores[key] = scores
        print(
            f"  alpha {summary['alpha']:+.4f}"
            f"  95% CI [{summary['ci'][0]:+.4f}, {summary['ci'][1]:+.4f}]"
            f"  ({len(summary['per_lab'])} labs, {time.perf_counter() - t0:.1f}s)"
        )

    print(f"\n[tfidf] TF-IDF floor (n_boot={TFIDF_N_BOOT}, see docstring)")
    dense, keep, vocab, capped = tfidf_arm(texts)
    dropped = int((~keep).sum())
    print(f"  vocabulary {vocab:,}" + (f" -> capped at {TFIDF_MAX_FEATURES:,}" if capped else ""))
    print(f"  dropped {dropped} zero-vector annotation(s) (non-lexical strings)")
    t0 = time.perf_counter()
    summary, scores = analyze_arm(episodes[keep], dense, TFIDF_N_BOOT)
    summary["gap"] = summary["alpha"] - nom.alpha
    results["tfidf"] = summary
    episode_scores["tfidf"] = scores
    print(
        f"  alpha {summary['alpha']:+.4f}"
        f"  95% CI [{summary['ci'][0]:+.4f}, {summary['ci'][1]:+.4f}]"
        f"  ({len(summary['per_lab'])} labs, {time.perf_counter() - t0:.1f}s)"
    )

    # --- does the choice of encoder change the *conclusions*? ---------------
    lab_rho: dict[str, float] = {}
    ep_rho: dict[str, float] = {}
    for a, b in itertools.combinations(ENCODER_ORDER, 2):
        labs_common = sorted(set(results[a]["per_lab"]) & set(results[b]["per_lab"]))
        if len(labs_common) >= 3:
            rho = spearmanr(
                [results[a]["per_lab"][l]["alpha"] for l in labs_common],
                [results[b]["per_lab"][l]["alpha"] for l in labs_common],
            ).statistic
            lab_rho[f"{a}|{b}"] = float(rho)
        eps_common = sorted(set(episode_scores[a]) & set(episode_scores[b]))
        rho = spearmanr(
            [episode_scores[a][e] for e in eps_common],
            [episode_scores[b][e] for e in eps_common],
        ).statistic
        ep_rho[f"{a}|{b}"] = float(rho)

    print("\n" + "=" * 78)
    print("ENCODER ROBUSTNESS")
    print("=" * 78)
    print(f"  {'encoder':<8} {'dims':>5} {'alpha':>8} {'95% CI':>19} {'D_o':>7} {'D_e':>7} {'gap':>8}")
    for key in ENCODER_ORDER:
        r = results[key]
        print(
            f"  {key:<8} {r['dims']:>5} {r['alpha']:>+8.4f} "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
            f"{r['d_o']:>7.4f} {r['d_e']:>7.4f} {r['gap']:>+8.4f}"
        )
    print(f"\n  paraphrase gap = semantic alpha - nominal alpha ({nom.alpha:+.4f})")
    print(f"  neural arms: {args.boot} bootstrap replicates; tfidf: {TFIDF_N_BOOT}")

    print("\n  rank agreement across encoders (Spearman rho)")
    print(f"  {'pair':<16} {'per-lab alpha':>14} {'per-episode':>12}")
    for a, b in itertools.combinations(ENCODER_ORDER, 2):
        pair = f"{a}|{b}"
        lab_s = f"{lab_rho[pair]:+.3f}" if pair in lab_rho else "   n/a"
        print(f"  {pair:<16} {lab_s:>14} {ep_rho[pair]:>+12.3f}")

    out = CACHE / "encoder_robustness.json"
    out.write_text(
        json.dumps(
            {
                "nominal_alpha": nom.alpha,
                "encoders": {
                    key: {"model": NEURAL.get(key, "tfidf"), **results[key]}
                    for key in ENCODER_ORDER
                },
                "pairwise": {
                    "lab_rank_spearman": lab_rho,
                    "episode_rank_spearman": ep_rho,
                },
                "tfidf_dropped_rows": dropped,
                "tfidf_vocab_size": vocab,
                "tfidf_capped_at": TFIDF_MAX_FEATURES if capped else None,
                "n_boot_neural": args.boot,
                "n_boot_tfidf": TFIDF_N_BOOT,
                "limit": args.limit,
            },
            indent=2,
        )
    )
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
