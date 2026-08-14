"""Weekend one: how much do DROID's three annotators agree with each other?

DROID paid crowdworkers to write natural-language instructions for its robot
episodes, collecting up to three independent descriptions each. The dataset
paper reports no inter-annotator agreement, no error rate, and no quality
validation for language. This script computes the missing number.

The annotations ship as a separate 12 MB JSON, so none of the 1.8 TB of video is
needed. Structure:

    "IRIS+7dfa2da3+2023-04-25-11h-42m-28s": {
        "language_instruction1": "Pour the contents of the bottle on the right into the sink",
        "language_instruction2": "Pour the contents of the larger container into the sink",
        "language_instruction3": "Pour the content of the bottle into the sink"
    }

Run:
    python scripts/droid_agreement.py                 # download + full analysis
    python scripts/droid_agreement.py --limit 2000    # quick smoke test first
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from vla_label_audit.crossmodal import effective_rank, instruction_space_report
from vla_label_audit.scalable import (
    alpha_nominal,
    alpha_semantic,
    bootstrap_alpha_semantic,
    per_unit_disagreement,
)

ANNOTATION_URL = (
    "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/"
    "aggregated-annotations-030724.json"
)
CACHE = Path("data")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, ~80 MB, fast on MPS


def fetch_annotations() -> dict:
    """Download the annotation JSON once, then read from disk forever after."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "droid_annotations.json"
    if not path.exists():
        print(f"downloading annotations (~12 MB) from\n  {ANNOTATION_URL}")
        try:
            urllib.request.urlretrieve(ANNOTATION_URL, path)
        except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
            raise SystemExit(
                f"download failed: {exc}\n\n"
                "Fallback, if you have the Google Cloud SDK:\n"
                "  gsutil -m cp gs://gresearch/robotics/droid_raw/1.0.1/"
                "aggregated-annotations-030724.json data/droid_annotations.json"
            ) from exc
    return json.loads(path.read_text())


def flatten(raw: dict, limit: int | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """One row per annotation. Returns (episode_ids, texts, annotator_slots).

    Empty and whitespace-only instructions are dropped rather than embedded --
    an empty string has no meaningful direction in embedding space, and
    counting it as disagreement would inflate the result.
    """
    episodes, texts, slots = [], [], []
    dropped_empty = 0
    for i, (episode, fields) in enumerate(raw.items()):
        if limit and i >= limit:
            break
        for slot in (1, 2, 3):
            text = (fields.get(f"language_instruction{slot}") or "").strip()
            if not text:
                dropped_empty += 1
                continue
            episodes.append(episode)
            texts.append(text)
            slots.append(slot)
    if dropped_empty:
        print(f"  dropped {dropped_empty:,} empty instruction slots")
    return np.array(episodes), texts, np.array(slots)


def embed(texts: list[str], tag: str) -> np.ndarray:
    """Sentence embeddings, cached to disk keyed by corpus size."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"embeddings_{tag}.npy"
    if path.exists():
        print(f"  using cached embeddings at {path}")
        return np.load(path)

    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  embedding {len(texts):,} sentences with {MODEL} on {device}")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    np.save(path, emb)
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only use the first N episodes")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap replicates")
    ap.add_argument("--top", type=int, default=15, help="worst episodes to print")
    args = ap.parse_args()

    raw = fetch_annotations()
    print(f"\nloaded {len(raw):,} episodes from the annotation file")

    episodes, texts, slots = flatten(raw, args.limit)
    n_ep = len(set(episodes.tolist()))
    print(f"  {len(texts):,} annotations across {n_ep:,} episodes")
    counts = np.bincount(np.unique(episodes, return_inverse=True)[1])
    for k in sorted(set(counts.tolist())):
        print(f"    {int((counts == k).sum()):,} episodes with {k} annotation(s)")

    emb = embed(texts, tag=f"{len(texts)}")

    print("\n" + "=" * 66)
    print("DO THE ANNOTATORS AGREE?")
    print("=" * 66)
    sem = alpha_semantic(episodes, emb)
    nom = alpha_nominal(episodes, texts)
    print(f"  semantic alpha (do they mean the same?)  {sem.alpha:+.4f}")
    print(f"  nominal alpha  (do they type the same?)  {nom.alpha:+.4f}")
    print(f"  paraphrase gap                           {sem.alpha - nom.alpha:+.4f}")
    print(f"\n  observed disagreement  {sem.observed:.4f}")
    print(f"  expected disagreement  {sem.expected:.4f}")
    print(f"  ({sem.n_units:,} pairable episodes, {sem.n_pairable:,} annotations)")

    print(f"\n  bootstrapping ({args.boot} replicates, resampling episodes)...")
    t0 = time.perf_counter()
    point, lo, hi = bootstrap_alpha_semantic(episodes, emb, n_boot=args.boot, seed=0)
    print(f"  semantic alpha = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   ({time.perf_counter()-t0:.1f}s)")
    print("\n  1.0 = perfect agreement | 0.0 = no better than random | <0 = worse than chance")

    print("\n" + "=" * 66)
    print("HOW MUCH VARIETY IS IN THE LANGUAGE?")
    print("=" * 66)
    uniq = len(set(texts))
    print(f"  unique instruction strings   {uniq:,} of {len(texts):,}  ({uniq/len(texts):.1%})")
    print(f"  effective rank of the space  {effective_rank(emb):.1f} of {emb.shape[1]} dims")
    # The similarity statistics need an N x N matrix, so they run on a sample.
    sample = np.random.default_rng(0).choice(len(texts), size=min(4_000, len(texts)), replace=False)
    rep = instruction_space_report(emb[sample])
    print(f"  mean nearest-neighbour sim   {rep['mean_nearest_neighbor_similarity']:.3f}  (4k sample)")
    print(f"  mean pairwise similarity     {rep['mean_pairwise_similarity']:.3f}  (4k sample)")

    print("\n" + "=" * 66)
    print("BY LAB")
    print("=" * 66)
    labs = np.array([e.split("+")[0] for e in episodes])
    rows = []
    for lab in np.unique(labs):
        m = labs == lab
        if len(set(episodes[m].tolist())) < 30:
            continue
        try:
            rows.append((lab, alpha_semantic(episodes[m], emb[m])))
        except ValueError:
            continue
    for lab, res in sorted(rows, key=lambda r: r[1].alpha)[:12]:
        print(f"  {lab:<22} alpha {res.alpha:+.4f}   ({res.n_units:,} episodes)")
    if len(rows) > 12:
        print(f"  ... {len(rows) - 12} more labs")

    print("\n" + "=" * 66)
    print(f"THE {args.top} EPISODES WHERE ANNOTATORS DISAGREED MOST")
    print("=" * 66)
    scores = per_unit_disagreement(episodes, emb)
    by_ep: dict[str, list[str]] = {}
    for e, t in zip(episodes.tolist(), texts):
        by_ep.setdefault(e, []).append(t)
    for ep, score in sorted(scores.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"\n  [{score:.3f}] {ep}")
        for t in by_ep[ep]:
            print(f"       - {t}")

    out = CACHE / "droid_agreement_results.json"
    out.write_text(
        json.dumps(
            {
                "n_episodes": int(sem.n_units),
                "n_annotations": int(sem.n_pairable),
                "alpha_semantic": sem.alpha,
                "alpha_semantic_ci": [lo, hi],
                "alpha_nominal": nom.alpha,
                "observed_disagreement": sem.observed,
                "expected_disagreement": sem.expected,
                "unique_string_fraction": uniq / len(texts),
                "effective_rank": effective_rank(emb),
                "per_lab_alpha": {lab: r.alpha for lab, r in rows},
                "worst_episodes": dict(sorted(scores.items(), key=lambda kv: -kv[1])[:200]),
            },
            indent=2,
        )
    )
    print(f"\n\nresults written to {out}")


if __name__ == "__main__":
    main()
