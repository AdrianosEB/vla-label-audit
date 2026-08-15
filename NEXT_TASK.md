# Next task: encoder robustness

**Read `CLAUDE.md` first if you haven't.** Do not change anything in `vla_label_audit/` — the
statistics modules are settled and tested. This is a scripts-and-analysis task.

## Why this matters

`alpha_semantic` measures agreement using cosine distance between sentence embeddings. Every
number we have — α = 0.8125, the per-lab ordering, the paraphrase gap — is computed in
`all-MiniLM-L6-v2`'s geometry. Nobody has checked whether those numbers describe **DROID** or
describe **MiniLM**.

If a different encoder returns 0.65, the absolute value is an artifact and the paper can only
make comparative claims. If several encoders land in a narrow band, the absolute value is
reportable. Either answer is publishable; not knowing is not.

Related: the conventional α ≥ 0.80 reliability threshold comes from *categorical* content
analysis. Its transfer to an embedding distance is an assumption, not a calibrated fact. That
is a second reason the comparative structure may matter more than the absolute number.

## What to build

Add `scripts/encoder_robustness.py`. Feel free to use subagents to run encoders in parallel.

**Encoders** — pick 4–5 spanning genuinely different families, sizes, and training recipes, not
four variants of the same thing. Something like:

- `all-MiniLM-L6-v2` (current baseline, 384-d, 22M)
- `all-mpnet-base-v2` (768-d, 110M — different architecture and objective)
- a BGE or GTE model (different training recipe entirely)
- one noticeably larger model, if it fits in 16 GB comfortably
- **a non-neural floor: TF-IDF cosine.** This one is important. It is the "no semantics at all"
  control, and it tells us how much of α is real semantic agreement versus lexical overlap that
  any bag-of-words method would find.

Cache per-encoder embeddings; the corpus is 125,276 sentences and re-embedding is the expensive
part.

## What to measure

For each encoder, on the full file:

1. **Semantic α with episode-clustered bootstrap CI.** The headline comparison.
2. **Per-lab α.** Then the number that actually decides the paper's structure:
   **Spearman rank correlation of the lab ordering between every pair of encoders.** If the
   ranking is stable while the absolute α moves, comparative claims are safe and absolute ones
   are not — which is exactly the distinction we need to resolve.
3. **The paraphrase gap** (semantic α minus nominal α). Nominal α is encoder-independent, so
   only the semantic side moves; report whether the gap survives everywhere.
4. **Expected disagreement `D_e` per encoder.** This is the denominator of α and it is a pure
   property of the encoder's geometry — some embedding spaces are far more anisotropic than
   others, and a low `D_e` mechanically inflates α. Worth reporting alongside, because if α
   tracks `D_e` across encoders that is a warning rather than a finding.
5. **Rank correlation of the per-episode disagreement scores** between encoders. Do the encoders
   agree about *which specific episodes* are the worst? If yes, the ranked worklist is a real
   artifact rather than one model's opinion.

## What I want back

- The α table across encoders, with CIs.
- A blunt verdict: **is the absolute number reportable, or only the comparisons?**
- Whether the TF-IDF floor is close to the neural encoders. If TF-IDF gets 0.75, most of what we
  are calling semantic agreement is lexical overlap, and that changes the story materially.
- Anything that surprised you.

## Rules

- Do not weaken or modify any existing test.
- Do not touch `vla_label_audit/` except to *add* a module if you genuinely need one; if you
  think you do, say why first.
- If an encoder does not fit in memory or takes absurdly long on MPS, drop it and say so rather
  than silently substituting something else.
- Report what the runs actually produced. If something failed, that is the finding.
