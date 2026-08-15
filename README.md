# vla-label-audit

**Do the natural-language labels in robot-learning datasets actually describe the trajectories
they are attached to?**

Vision-language-action models are known to largely ignore the language they are given.
[RoboSemanticBench](https://arxiv.org/html/2606.02277) reports 80–100% grasp success but only
2–21% *commanded-task* success. The field reads that as an architecture problem. Before
accepting that, it is worth checking something cheaper: **were the training labels correct in
the first place?** Nobody had measured it. This repository measures it.

The short answer, on DROID: the labels are **not** unreliable — annotators agree on meaning at
Krippendorff's α = 0.81. What they almost never do is agree on *wording*. That gap is the
finding.

---

## Headline results

Measured on the full DROID annotation file: **50,092 episodes, 125,276 annotations**, of which
37,592 episodes are triple-annotated.

| Quantity | Value |
|---|---|
| **Semantic α** (all-MiniLM-L6-v2) | **0.8125**, 95% CI [0.8107, 0.8140] |
| Nominal α (exact string match) | 0.0536 |
| **Paraphrase gap** (semantic − nominal) | **0.7589** |
| Lexical floor (TF-IDF cosine) | 0.6310, 95% CI [0.6285, 0.6334] |
| Unique instruction strings | 64,516 / 125,276 (51.5%) |

To our knowledge this is the **first inter-annotator agreement measurement for natural-language
annotation in a robot-learning dataset.** (See *Limitations* — the novelty search predates
submission and should be re-run.)

### 1. The number is a property of the data, not of one embedding model

Every α here is a Krippendorff's α whose difference function is cosine distance between
sentence embeddings — so the obvious objection is that it measures the encoder, not the
annotators. It does not. Five encoder arms were run over the full corpus; the four neural
encoders land in a band **0.012 wide**:

| Encoder | dims | Semantic α | 95% CI |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.8125 | [0.8107, 0.8140] |
| all-mpnet-base-v2 | 768 | 0.8218 | [0.8200, 0.8235] |
| gte-base | 768 | 0.8095 | [0.8078, 0.8110] |
| bge-large-en-v1.5 | 1024 | 0.8165 | [0.8146, 0.8180] |
| *TF-IDF (deliberate lexical floor)* | 2442 | *0.6310* | *[0.6285, 0.6334]* |

Report α ≈ 0.81 **with the band, never with a single arm's CI** — encoder choice contributes
roughly 4× the sampling uncertainty, and the arms differ significantly from one another even
while agreeing practically. The claim is scoped to contrastively-trained sentence-encoder
families; no decoder-LM embedder was tested.

α is also not an artifact of embedding geometry: expected disagreement `D_e` spans 20× across
the arms (0.040 to 0.818) while neural α moves 0.012, and the most anisotropic encoder produces
the *lowest* α — the opposite of mechanical inflation.

### 2. The paraphrase gap

Annotators overwhelmingly mean the same thing (0.81) and almost never say it the same way
(0.054). The gap survives every encoder (0.756–0.768). It is not a formatting artifact:
lowercasing moves nominal α to 0.0538, additionally stripping punctuation to 0.0577. On
episodes containing no identical-string pairs at all, nominal α is ≈0.000 while semantic α is
still 0.79–0.80.

Contrast RT-1, which has <2% unique instructions and 49 unique words across 3.7M sentences.
Two opposite pathologies: **RT-1 too templated to teach language, DROID too varied to give
repeated signal per phrasing.**

### 3. How much of "semantic agreement" is just shared vocabulary?

This is the decomposition the TF-IDF arm exists to provide, and it cuts both ways:

- **0.63 is reachable by vocabulary overlap alone** — a bag-of-words method with no notion of
  meaning. Under every lexical variant tried (char 3–5-grams reach 0.6697 at best; word
  1–2-grams 0.5441) the floor stays ≥0.14 below the neural band.
- **≈0.18 requires meaning.** That increment is real and robust, but the honest framing is
  "most of the agreement is lexical, and a substantial minority is not" — not "agreement is
  semantic, full stop."

### 4. Site-level variation exceeds sampling noise

The ten collection sites span **0.767 to 0.887** in per-site α — an order of magnitude wider
than any individual site's confidence interval (~0.011 wide). Something real differs between
labs.

Adversarial verification sharpened what may be claimed about *which* sites:
**GuptaLab is robustly the highest** (P(argmax) ≈ 1.000 in every encoder arm, CI disjoint from
the runner-up, surviving tie-rate, annotation-length and sample-size confound checks). The low
end is a **{CLVR, RAIL} cluster**, not a single identifiable site — P(CLVR is the true minimum)
is only 0.73, and the ordering flips under a tie-removal sensitivity check. Agreement across
five encoder arms is *not* five independent confirmations, because all arms score the same
episodes.

---

## Two methodological findings that generalize beyond this dataset

These are the transferable results, and both are warnings.

### Per-episode embedding rankings are ~50% encoder-dependent at the tail

Aggregate embedding statistics over a corpus are robust. **Per-episode rankings built from the
same embeddings are not.** Between encoders that agree on the corpus-level α to within 0.012,
the top-200 "worst episode" lists share only **47–57%** of their members, and ordering *within*
the tail is nearly uncorrelated (tail-only Spearman 0.06–0.31). What does transfer is
neighbourhood membership: 93–99% of one encoder's top-200 falls inside another's top-2000.

The general form: **an aggregate statistic being encoder-robust says nothing about the tail of
a per-item ranking being encoder-robust.** Any pipeline that ranks items by embedding
disagreement and consumes a top-k list — data-cleaning worklists, active-learning queues,
"worst examples" tables — inherits roughly 50% churn at the tail from encoder choice alone,
and that churn is invisible if only the aggregate is checked. This applies to the
`worst_episodes` list this project itself produced in its first session, which is
single-encoder; treat it as a sample from a high-disagreement pool, not as a ranking.

### Neighbourhood-disagreement detectors are blind to *correlated* label errors

A natural way to hunt for mislabeled episodes is to flag those whose label disagrees with the
labels of their nearest neighbours in embedding space. That method has a structural blind spot,
and it is the one that matters in practice.

When a group of episodes shares the *same* wrong label — one annotator with a consistent
misconception, one template applied to the wrong batch, one scripted relabel — the corrupted
episodes' neighbours carry that same wrong label, so neighbourhood disagreement is **zero by
construction**. Under injected correlated corruption of exactly this kind, detection falls to
**chance: ROC AUC 0.487–0.511, precision@50 of 0.03–0.19.**

This is a property of the method family, not of any encoder, distance metric, or ensemble. It
applies wherever mislabeling is batched rather than independent — which is how real annotation
pipelines fail. **The method finds isolated label errors and is blind to systematic ones.**

---

## Limitations, stated up front

- **The α ≥ 0.80 reliability convention comes from categorical content analysis.** Its transfer
  to a continuous cosine-distance metric is an assumption, not a calibrated fact. Treat "0.81
  clears the threshold" as suggestive, and prefer the comparative structure to the absolute
  number.
- **Encoder choice contributes ~4× the sampling uncertainty.** Never quote a single arm's CI as
  the uncertainty on α ≈ 0.81.
- **The band is scoped to contrastive sentence encoders.** It rules out "MiniLM artifact", not
  "sentence-encoder-family artifact".
- **A cross-modal audit was attempted and did not validate.** See below — the negative result is
  reported in full rather than dropped.
- **Degenerate agreement inflates α.** 103 of the 115 episodes containing a non-answer have all
  three annotators writing "No action". These are perfectly-agreeing units with no instruction
  to agree about. They are kept in the headline, reported as a robustness row, and the episode
  list is released — silently dropping them would measure a corpus nobody trains on.
- **The DROID annotation file is sorted** (all triple-annotated episodes first, then all
  singles), so `--limit N` is never a random sample and its output must not be quoted as an
  estimate of the full-file number.
- **No human base-rate audit has been done.** Nobody has hand-checked a random sample of DROID
  episodes against their instructions. Without that denominator, no detector's precision can be
  honestly evaluated — including any built here.
- **The novelty claim above predates submission.** This area moves monthly; the search should be
  re-run before any preprint.

## The cross-modal audit: a negative result

A second stage tried to move from "do annotators agree with each other" to "does the label
describe what the camera saw", using LIBERO (1,693 episodes, 546,930 images embedded with
DINOv2 ViT-S/14 and CLIP ViT-B/32). **It did not validate, and the reason is about the corpus,
not about the idea.**

LIBERO has 40 instructions in bijection with 40 `task_index` classes and near-bijection with
scene identity. Under that structure, "language neighbourhood" reduces exactly to "`task_index`
equality class", and the whole cross-modal apparatus collapses into a lookup. **LIBERO cannot
discriminate the hypothesis under test from the null "the dataset has 40 blocks."** It is also
confounded on a second axis: its instructions are the task definitions that generated its
scripted simulated demonstrations, so its true label-noise rate is ≈0 and there is nothing real
to find. Precision can only be assessed against planted errors, and the planted error that is
natural to inject is precisely the trivially-detectable case.

**LIBERO therefore cannot validate this class of method.** The right conclusion is that the
validation corpus was structurally unsuitable — not that cross-modal auditing has been shown to
fail in general. DROID, with 64,516 unique strings and no `task_index`, has neither pathology;
whether the approach works there is **untested**.

Every quantitative result from that stage is specific to LIBERO's block structure and is
deliberately **not** reported here as a finding about DROID or about the method in general. The
full record, including what was refuted and why, is in [`results/stage-B.md`](results/stage-B.md).
The one result from that stage that *does* generalize — correlated-error blindness — is stated
in the section above.

---

## Reproducing this

Requires Python ≥3.10 (a 3.11 virtualenv is what this was developed against). Embeddings are
cached to `data/`; nothing under `data/` is committed.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # expect 50 passed
```

**Stage A — agreement and encoder robustness.** The annotation file (~12 MB) downloads
automatically on first run, with a `gsutil` fallback printed if the download fails.

```bash
# baseline agreement on the full corpus (semantic + nominal alpha, per-lab, worst episodes)
.venv/bin/python scripts/droid_agreement.py

# annotation defect classes (truncation, junk, non-answers, punctuation)
.venv/bin/python scripts/annotation_quality.py

# encoder robustness: embed once per encoder (resumable, cached), then analyze
.venv/bin/python scripts/encoder_robustness.py --embed-only --encoder mpnet
.venv/bin/python scripts/encoder_robustness.py --embed-only --encoder gte
.venv/bin/python scripts/encoder_robustness.py --embed-only --encoder bge-l
.venv/bin/python scripts/encoder_robustness.py --analyze
```

Use `--limit N` for a fast smoke run, but **never quote a `--limit` number as a corpus
estimate** — the file is sorted, so it is not a random sample.

**Cross-modal stage (LIBERO).** Downloads ~1.9 GB and embeds 546,930 images (~87 min on an
M2 Pro across both encoders).

```bash
.venv/bin/python scripts/libero_embed.py --check-mps --encoder dinov2   # MPS/CPU agreement check
.venv/bin/python scripts/libero_embed.py --encoder dinov2
.venv/bin/python scripts/libero_embed.py --encoder clip
.venv/bin/python scripts/libero_crossmodal.py --build-views
.venv/bin/python scripts/libero_crossmodal.py --analyze
```

## Vector search

A FAISS index over the cached embeddings, for exploring the corpus rather than for producing
any number in the analysis. Three modes: text → DROID episodes, text → LIBERO frames
(CLIP text tower into the CLIP image space), and episode → similar episodes.

```bash
.venv/bin/python scripts/build_index.py --which all     # ~5 s, writes data/index/ (gitignored)

.venv/bin/python scripts/search.py --text-to-episodes "pick up the mug and put it in the sink"
.venv/bin/python scripts/search.py --text-to-frames "a robot arm picking up a black bowl"
.venv/bin/python scripts/search.py --episode-similar "PennPAL+acda9df3+2023-06-25-18h-18m-03s"
.venv/bin/python scripts/search.py --text-to-episodes "..." --ivf --nprobe 32   # approximate
```

DROID results carry each annotation's lab and its episode's disagreement score, so a search
doubles as a way to look at how differently three annotators described the same clip.

**Exact is the default; approximate is opt-in.** The analysis uses exact search everywhere
because at this scale brute force costs milliseconds and exactness pre-empts the "did ANN
recall cause that result?" objection before a reviewer can raise it — whereas the interactive
tool can trade a little recall for a ~10–140× latency drop, which is why the benchmark below
scores IVF *against* exact rather than replacing it.

| index | method | recall@10 | median ms | p95 ms |
|---|---|---|---|---|
| droid_text (125,276 × 384, nlist 350) | **exact** | 1.000 | 3.992 | 4.087 |
| droid_text | ivf nprobe=1 | 0.8640 | 0.027 | 0.054 |
| droid_text | ivf nprobe=8 | 0.9725 | 0.117 | 0.157 |
| droid_text | ivf nprobe=32 | 0.9905 | 0.395 | 0.455 |
| droid_text | ivf nprobe=64 | 0.9935 | 0.817 | 0.908 |
| libero_frame (273,465 × 512, nlist 520) | **exact** | 1.000 | 10.792 | 11.124 |
| libero_frame | ivf nprobe=1 | 0.7310 | 0.047 | 0.065 |
| libero_frame | ivf nprobe=8 | 0.9905 | 0.212 | 0.280 |
| libero_frame | ivf nprobe=32 | 1.0000 | 0.765 | 0.958 |
| libero_frame | ivf nprobe=64 | 1.0000 | 1.524 | 1.743 |

200 random queries drawn from the indexed vectors, k=10, single-threaded, exact
`IndexFlatIP` as ground truth; recall is top-10 set overlap. The two 1.0000 rows are exact
(2000/2000), not rounded — LIBERO's nearest neighbours are usually temporally adjacent frames
of the same episode, which land in the same IVF cell. Raw output in `data/index/benchmark.json`.

Two implementation notes that will bite anyone reusing this:

- **`torch` and `faiss-cpu` cannot share one interpreter here** — both ship their own libomp on
  macOS and whichever enters a parallel region second dies with a bare SIGSEGV, no traceback.
  `faiss.omp_set_num_threads(1)`, `torch.set_num_threads(1)` and `KMP_DUPLICATE_LIB_OK` do not
  help. `search.py` therefore embeds each query in a child process that imports torch and never
  faiss, while the parent imports faiss and never torch.
- **`CLIPModel.get_text_features` is a silent-corruption trap on transformers 5.15**: it returns
  a pooled output whose `pooler_output` is the *pre*-projection hidden state, which is also
  512-d for ViT-B/32 — so it would index cleanly and return confident nonsense. Use
  `CLIPTextModelWithProjection` → `.text_embeds`, the true counterpart of the cached
  `image_embeds`.

Text → frame retrieval on LIBERO is visibly weak (queries return plausible-but-wrong episodes
at similarities clustered near 0.36). The plumbing is verified — correct projected tower, both
sides L2-normalised, IVF recall 1.000 against exact — so this reflects CLIP's alignment on
simulated robot footage. It is an observation about this corpus and this encoder, not a
measurement, and nothing in the audit depends on it.

## How this repository is organised

| Path | What it is |
|---|---|
| `vla_label_audit/` | The statistical core. `agreement.py` is the naive reference implementation; `scalable.py` computes the identical quantity in O(n·d²) via a closed form (the textbook N×N difference matrix would be 90 GB at DROID scale). `crossmodal.py` holds the cross-view diagnostics. |
| `scripts/` | Analyses. Nothing here is imported by the package. |
| `results/` | Stage reports, including what was refuted and what could not be verified. |
| `tests/` | 50 tests. Several are correctness claims, not smoke checks — one pins the fast α path to the naive one at 1e-10, one guards a bug that allocated 59 GB, one validates an analytic power formula against simulation. |
| `data/` | Caches and downloads. Gitignored in its entirety. |

Every finding here was produced by one process and then attacked by independent adversarial
verifiers instructed to refute it and to default to "refuted" when uncertain. Research
decisions — study design, corpus choice, what to claim and at what scope — were made by the
author at explicit review gates; two of the project's own hypotheses were killed that way.
Findings that did not survive are recorded alongside those that did; see `results/`.

## License

MIT — see [LICENSE](LICENSE).
