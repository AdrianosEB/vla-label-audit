# Stage A — encoder robustness

Full DROID corpus: 50,092 episodes, 125,276 annotations, 37,592 triple-annotated.
Analysis: `scripts/encoder_robustness.py` → `data/encoder_robustness.json` (limit: none,
n_boot 300 neural / 50 TF-IDF). Test suite: 50/50 passing before and after.
Verified adversarially by three independent verifiers (statistical validity; confounds;
independent reproduction), each instructed to refute and to default to refuted when uncertain.

## Numbers

| Encoder | dims | Semantic α | 95% CI | D_o | D_e | Gap vs nominal |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.8125 | [0.8107, 0.8140] | 0.0890 | 0.4749 | 0.7589 |
| all-mpnet-base-v2 | 768 | 0.8218 | [0.8200, 0.8235] | 0.0736 | 0.4132 | 0.7682 |
| gte-base | 768 | 0.8095 | [0.8078, 0.8110] | 0.0076 | 0.0400 | 0.7559 |
| bge-large-en-v1.5 | 1024 | 0.8165 | [0.8146, 0.8180] | 0.0267 | 0.1454 | 0.7628 |
| TF-IDF (floor) | 2442 | 0.6310 | [0.6285, 0.6334] | 0.3017 | 0.8176 | 0.5774 |

Nominal (exact-string) α = 0.0536, encoder-independent. Neural band width 0.0123.
Per-lab αs, pairwise lab-rank Spearman (0.79–0.99), and per-episode rank Spearman
(0.88–0.94 neural pairs) are in the JSON.

## Gate A answers

1. **Is the absolute α reportable?** Yes, as ≈0.81 with an honest uncertainty statement.
   Four genuinely different neural encoders span 0.8095–0.8218. Two obligations attach:
   (a) encoder choice contributes ~4× the sampling uncertainty, so no single per-encoder CI
   may be quoted as the uncertainty of "≈0.81" — report the band; the encoders differ
   *significantly* from one another (mpnet and gte CIs are disjoint) while agreeing
   practically. (b) All four are contrastively trained sentence encoders; this rules out
   "MiniLM artifact", not "sentence-encoder-family artifact".
2. **Is the per-lab ranking stable?** No — and even the previously drafted "extremes are
   robust" claim was half-refuted in verification (see below). What is safe:
   **GuptaLab is uniquely highest** (P(argmax) ≈ 1.000 in every arm, CI disjoint from the
   runner-up everywhere, survives tie-removal and length confound checks), and
   **{CLVR, RAIL} form the low cluster**. CLVR-uniquely-lowest is *not* supportable:
   P(CLVR = argmin) = 0.73 on the headline encoder, and the CLVR/RAIL ordering flips under
   a tie-removal sensitivity check. Cross-lab claims must be worded at cluster resolution.
3. **Where does TF-IDF land?** 0.6310 — well below the neural band (gap ≥ 0.14 under every
   lexical preprocessing variant tried: char 3–5-grams reach 0.6697 at best). A substantial
   share of measured agreement is genuinely semantic. But 0.63 is itself a high floor:
   annotators reuse vocabulary heavily, and the paper should say both things.

## What survived verification

- **Cross-encoder stability of α (claim 1)** — 3 verifiers, 0 refutations. Every point
  estimate reproduced independently (to ~1e-14 from cached embeddings; MiniLM arm
  bit-identical to the stored Session-1 baseline). Stability is not manufactured by
  identical-string ties: ties are only 5.4% of within-episode pairs; on the 32,555
  episodes with zero internal ties the band is 0.7914–0.8040, width 0.0126 —
  unchanged. n_boot=300 gives CI-endpoint Monte-Carlo noise of ±0.0001–0.0002.
- **TF-IDF floor (claim 2)** — 3 verifiers, 0 refutations. Neural−lexical separation is
  ~140 bootstrap SDs; robust to preprocessing; the 8 dropped zero-vector rows are 0.006%
  of the corpus. Caveat: unigram TF-IDF is a weak lexical model, so 0.63 is a floor for
  lexical agreement, possibly slightly low.
- **α does not track anisotropy (claim 3)** — 3 verifiers, 0 refutations. D_e spans 20×
  (0.040–0.818) while neural α moves 0.012; gte is the *most* anisotropic and gives the
  *lowest* neural α — the opposite of mechanical inflation. fp16 storage of the gte cache
  changes α by <1e-5 (tested by casting every cache to fp16). Limitation: four
  observational points; cannot exclude an inflation common to all cosine-embedding
  metrics. Proposed follow-up if a reviewer pushes: whitening/ABTT intervention.
- **Paraphrase gap (claim 6)** — 3 verifiers, 0 refutations. Nominal α reproduced to the
  last digit by a from-scratch implementation (0.053604879526975235). Gap 0.756–0.768
  across encoders. Not a case/punctuation artifact (lowercasing moves nominal α to 0.0538;
  +punctuation-stripping to 0.0577). On tie-free episodes nominal α ≈ 0.000 while semantic
  α is still 0.79–0.80 — the gap is not driven by duplicate strings.

## What did not survive as originally worded

- **"The ranked worst-episode list is a property of the data" (claim 4)** — refuted by the
  statistical-validity verifier; the confounds verifier reached the same substance.
  The 0.88–0.94 global Spearman is dominated by the unremarkable bulk of 37,592 episodes.
  The actual top-200 worklists overlap only 47–57% between neural encoders, and
  within-tail ordering is nearly uncorrelated (tail-only Spearman 0.06–0.31).
  **What survives, restated:** worklist membership at neighborhood resolution — 93–99% of
  any encoder's top-200 lies inside another's top-2000 (top 5%), median cross-encoder rank
  of a top-200 episode is 152–220 of 37,592, and the result holds on tie-free episodes.
  Consequence: any released suspect list must be presented as a high-disagreement *pool*,
  not a precise ranking, and Stage B's "top ~50 suspects" must be drawn/validated with
  this instability in mind (e.g. intersect or union across encoders, state the resolution).
- **"CLVR lowest and GuptaLab highest — the extremes are robust" (claim 5)** — CLVR half
  refuted by 2 of 3 verifiers. "Lowest in all five arms" is not five independent
  confirmations: all arms score the same episodes, so sampling noise is shared, and
  P(CLVR truly lowest) is bounded by the weakest arm's ~0.73. GuptaLab-highest survived
  every attack, including tie-rate, string-uniqueness, annotation-length, and
  sample-size confounds. Supportable wording: GuptaLab uniquely on top; CLVR and RAIL
  jointly form the low extreme.

## Could not verify

- **Bootstrap CIs as stored** — point estimates and the MiniLM CI (matches the stored
  baseline bit-for-bit) were reproduced; the 300-rep full-corpus bootstrap for the other
  arms was not re-run end-to-end (hours of compute). Endpoint-stability was instead
  verified empirically on subsamples (MC noise ≤ ~10% of CI width; ±0.0004–0.0005 for the
  n_boot=50 TF-IDF arm, whose 4th decimal is therefore soft).
- **Encoder-family generality** — no non-contrastive (e.g. decoder-LM) embedding model was
  run; 16 GB and time budget. The band claim is scoped to sentence-encoder families.

## Surprises

- gte-base's geometry is ~12× more anisotropic than MiniLM's (D_e 0.040 vs 0.475), yet its
  α lands mid-band. Good news for the method; worth a sentence in the paper.
- The lexical floor is high in absolute terms (0.63; 0.67 with char n-grams). The story is
  not "agreement is semantic, full stop" — it is "roughly 0.63 of agreement is reachable by
  vocabulary overlap alone and the remaining ~0.18 requires meaning."
- Tie-removal lowers α by ~0.018 uniformly across encoders — a clean robustness row for
  the paper.

## Premise check (Gate A)

Verdict: **WEAKENED** — fresh premise-checker, minimal context. Deciding number:
the 47–57% top-200 overlap across encoders in the ranked worst-episode lists.

The checker split the premise in two. *"The paraphrase gap is a property of DROID, not
the encoder"* **holds** — the 0.012 neural band, the lexical floor ≥0.14 lower, and α's
indifference to a 20× swing in D_e; even the encoder-family caveat is blunted by the
TF-IDF floor, since a semantics-free method already finds 0.63 of the agreement. What is
**weakened** is *"…so Stage B, which relies on embedding-distance methods being
trustworthy, can proceed"*: embedding distances proved trustworthy in aggregate but not
at per-episode ranking granularity — which is exactly the granularity Stage B's "rank by
cross-modal disagreement and inspect the top ~50" uses — and cross-modal text–image
spaces will likely be *less* stable than the four same-modality encoders tested here.

**Required revision to the Stage B brief before it runs:** do not draw the suspect list
from any single encoder's ranking. Use a consensus/ensemble rank or an intersection of
top-N lists across at least two vision-language encoders; treat neighborhood membership
(93–99% top-200-within-top-2000) as the operative guarantee ("this episode is in the bad
tail", not "this episode is 37th worst"); and budget the manual inspection expecting
roughly half of any single-model top-50 to be encoder-specific noise.
