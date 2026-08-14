# vla-label-audit

**Do robot-learning datasets say what they think they say?**

Vision-language-action models are known to largely ignore the language they are given.
[RoboSemanticBench](https://arxiv.org/html/2606.02277) finds models grasping successfully 80–100%
of the time while completing the *commanded* task only 2–21% of the time, with normalized
semantic grounding near or below zero — they pick targets close to randomly.

The field reads this as an architecture problem. Nobody has checked whether the labels those
models trained on were correct in the first place.

## Why nobody has checked, and why they can now

[Wanna et al. (2026)](https://arxiv.org/html/2601.03136v1) measured the *text* of embodied-AI
datasets and found under 2% of instructions are unique, and that RT-1 uses **49 unique words
across 3.7M+ sentences**. They state explicitly that cross-modal alignment and annotation
correctness are outside their scope, and name detecting "inconsistencies between commands and
the corresponding trajectories" as future work.

Meanwhile [DROID](https://droid-dataset.github.io/) collected **up to three independent
crowdsourced instructions per episode** — the published annotation file carries on the order of
50,000 entries — and reported no inter-annotator agreement number, no error rate, and no
quality validation for language at all. (The paper claims coverage of 95% of successful
episodes; the released file is smaller than that implies, which is itself worth reporting.)

The measuring instrument is sitting in a public dataset, unused.

## What this package does

| Module | Question it answers |
|---|---|
| `agreement.py` | Do independent annotators describe the same episode the same way? |
| `crossmodal.py` | Does the label describe the trajectory it is attached to? |
| `noise.py` | What does a given rate of label noise actually cost a policy? |

### Agreement on free text

Standard agreement statistics assume categorical labels. These are sentences — "pick up the
red mug" and "grab the mug" agree; "move the arm left" does not, and no categorical statistic
can see the difference.

Krippendorff's alpha is defined over an *arbitrary* difference function, so supplying cosine
distance between sentence embeddings turns it into a measure of whether annotators described
the same behaviour rather than typed the same string. Reporting semantic alpha next to nominal
alpha isolates paraphrase from real disagreement — the gap between them *is* the paraphrase
rate.

```python
from vla_label_audit import krippendorff_alpha, cosine_distance_matrix, exact_match_distance_matrix

semantic = krippendorff_alpha(episode_ids, cosine_distance_matrix(annotation_embeddings))
nominal  = krippendorff_alpha(episode_ids, exact_match_distance_matrix(annotation_texts))
```

### Cross-modal audit

One index over episodes with three aligned views — what the camera saw, what the arm did, what
the label says — and every question becomes a nearest-neighbour query.

- `neighborhood_disagreement` — if an episode's *behavioural* neighbours all carry a different
  label than it does, either the label is wrong or the episode is genuinely rare. Both are
  worth surfacing. This is confident learning transplanted to free-text labels over
  trajectories, and it needs no ground truth.
- `neighborhood_overlap` / `rank_correlation_across_views` — corpus-wide: are visual neighbours
  language neighbours at all?
- `cca_alignment` — is there *any* linear map under which vision and language correspond?
  Canonical correlations near zero is a far stronger negative than low local overlap, and CCA
  is invariant to invertible linear reparameterization, so one encoder scaling its outputs
  differently cannot fake it (tested).
- `effective_rank` — a corpus advertising 160,000 tasks whose instruction embeddings span 8
  directions does not have 160,000 tasks. It has 8 templates and a lot of paraphrase.

### From description to causation

Measuring noise is descriptive. The claim worth making is what the noise *costs*, and nobody
can run that counterfactual — there is no clean version of DROID to compare against. So go the
other way: inject known noise, train at each level, fit the degradation curve, then use it to
convert a measured rate into a predicted cost.

Three noise modes, because the literature conflates them: `swap` (pipeline bugs — preserves
the label distribution), `shuffle` (the upper bound on damage), and `paraphrase` (the control
that separates real label noise from mere lexical variation).

## Quickstart

```bash
pip install -e .
pytest -q                            # 50 tests
python scripts/demo_synthetic.py     # end-to-end on a corpus with planted noise
python scripts/droid_agreement.py    # the real thing: DROID's annotator agreement
```

`scripts/droid_agreement.py` needs no video. DROID publishes its language annotations as a
separate **12 MB JSON**, so the entire first result comes from a file you can download over
coffee. Pass `--limit 2000` for a smoke test before the full run.

At corpus scale the textbook formulation of alpha would need a 150,000 x 150,000 distance
matrix — **90 GB**. `scalable.py` avoids it: observed disagreement only involves within-episode
pairs, and expected disagreement has a closed form under squared cosine distance
(`sum_ij (1-s_ij)^2 = n^2 - 2||sum_i x_i||^2 + ||X^T X||_F^2`), which costs O(n d^2) time and a
384x384 matrix of memory. Exact, not approximate — the tests pin it against the naive
implementation to 1e-10. Full DROID scale runs in about 4 seconds.

The demo builds a corpus with a **known 18.7% of labels deliberately corrupted** and checks
the audit can find them. It can:

```
semantic alpha  0.9991      string alpha  0.0000     <- paraphrase, not disagreement
alignment as-labelled: overlap 0.120, MI  3.57 nats
alignment if clean:    overlap 0.187, MI 16.80 nats  <- noise is visible in the geometry
top 10% most suspect: precision 100.0%, recall 53.6%
top 20% most suspect: precision  80.8%, recall 86.6%   (random baseline 18.7% -> 4.33x lift)
```

## Status

Statistical core complete and tested. No real-dataset results yet — nothing here should be
read as a finding about DROID or any other corpus.

## License

MIT.
