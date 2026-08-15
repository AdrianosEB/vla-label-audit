"""What is actually wrong with the text in DROID's annotation file?

`droid_agreement.py` answers "do the annotators agree". This script answers the
prior question: are the annotations well-formed sentences at all? Agreement
computed over a corpus that silently contains truncated fragments and "N/A"
placeholders is measuring something other than what it claims to.

Four defect classes are counted, each defined narrowly enough to be auditable:

* **truncated** -- cut off mid-word ("Turn on the kett", "Close the cof").
* **no terminal punctuation** -- reported for completeness, but see below: at
  ~91% this is the house style, not a defect.
* **non-answer** -- the annotator explicitly declined ("N/A", "Unsure",
  "No action").
* **junk** -- non-linguistic ("+++++++", a bare "g").

The headline question this script was built to settle is whether truncation is
an *annotation-tool* artefact -- a hard character limit silently cutting text.
That predicts (a) a spike in the length histogram at the cap and (b) truncated
annotations clustering at one length. Both are tested, plus association with
lab, with collection date, and with how long the *other* annotators on the same
episode wrote. Note that the truncation detector deliberately uses no episode-level
signal, so the sibling-length test is not circular.

Finally, semantic alpha is recomputed with each class excluded in turn, so the
cost of leaving the junk in is stated as a number rather than assumed.

Run:
    python scripts/annotation_quality.py              # full file
    python scripts/annotation_quality.py --limit 5000 # quick pass
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from droid_agreement import CACHE, corpus_tag, embed, fetch_annotations, normalize  # noqa: E402

from vla_label_audit.scalable import alpha_semantic, bootstrap_alpha_semantic  # noqa: E402

WORD = re.compile(r"[a-z]+")
TERMINAL = frozenset(".!?")
SYSTEM_DICT = Path("/usr/share/dict/words")

# Whole-string non-answers. Matched against the lower-cased, whitespace-collapsed
# text, anchored, so an instruction that merely *contains* "none" is untouched.
NON_ANSWER = re.compile(
    r"^(?:"
    r"n/?a|null|none|nan|nil|"
    r"unsure|not sure|unclear|unknown|idk|i don'?t know|can'?t tell|cannot tell|"
    r"no action|not action|no motion|no movement|no task|nothing|nothing happens|"
    r"no instruction|invalid|blank|empty"
    r")[.!?]?$"
)


def load_rows(raw: dict, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every annotation slot that holds text, normalised but not filtered.

    Deliberately *not* `droid_agreement.flatten`: that function drops empties,
    which is right for agreement and wrong here, since the point is to count what
    gets dropped. Absent slots (12,500 episodes carry only `instruction1`) are
    skipped -- a slot nobody was asked to fill is not a defective annotation.
    """
    episodes, slots, texts = [], [], []
    for i, (episode, fields) in enumerate(raw.items()):
        if limit and i >= limit:
            break
        for slot in (1, 2, 3):
            value = fields.get(f"language_instruction{slot}")
            if value is None:
                continue
            text = normalize(value)
            if not text:
                continue
            episodes.append(episode)
            slots.append(slot)
            texts.append(text)
    return np.array(episodes), np.array(slots), texts


def load_dictionary() -> set[str]:
    """The system word list, used only to *veto* truncation calls.

    /usr/share/dict/words is far too sparse to detect truncation on its own --
    it lacks "box", "countertop", "laptop" -- so it is never used to mark a word
    as broken. It is used in one direction only: if a word IS listed, it is a
    real word, so "stop" and "diagonal" cannot be mistaken for cut-off
    "stopper" and "diagonally".

    The veto applies only to tokens of four characters or more (see
    `flag_truncated`). This list carries every single letter as an entry plus a
    long tail of obscure short words -- "ta", "po", "gar" and "pac" are all in
    it -- so vetoing on short tokens discards the clearest truncations in the
    file ("in the clear b", "the silver bot").
    """
    if not SYSTEM_DICT.exists():
        return set()
    with SYSTEM_DICT.open(encoding="utf-8", errors="ignore") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


def flag_truncated(texts: list[str], words: set[str]) -> np.ndarray:
    """Mid-word truncation, judged against the corpus's own vocabulary.

    A truncated tail like "kett" is rare as a complete token yet is a prefix of
    tokens that are common ("kettle"). That ratio is the signal. Building the
    vocabulary from DROID itself rather than an external dictionary keeps the
    test in-domain, which matters when the domain is 2,500 words of tabletop
    manipulation vocabulary.

    Three guards keep precision high, which is what matters when the base rate
    is a handful in 125,000: the annotation must not end in terminal punctuation,
    the tail must appear at most twice as a whole word, and a tail of four or
    more characters must not be a real dictionary word.

    The residual ambiguity is genuine and irreducible from text alone: "Move the
    mic" is flagged because "mic" occurs once as a whole word against ~1,000
    occurrences of longer "mic-" words, but it could be a complete instruction
    about a microphone. The flagged set is small enough that the script prints
    all of it for inspection rather than asking to be trusted.
    """
    vocab: collections.Counter = collections.Counter()
    for text in texts:
        vocab.update(WORD.findall(text.lower()))

    extensions: collections.Counter = collections.Counter()
    for word, count in vocab.items():
        for cut in range(1, len(word)):
            extensions[word[:cut]] += count

    flags = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        if text[-1] in TERMINAL:
            continue
        tail = WORD.findall(text.lower())
        if not tail:
            continue
        tail = tail[-1]
        if len(tail) >= 4 and tail in words:
            continue
        if vocab[tail] <= 2 and extensions[tail] >= 20:
            flags[i] = True
    return flags


def flag_no_terminal_punct(texts: list[str]) -> np.ndarray:
    return np.array([t[-1] not in TERMINAL for t in texts], dtype=bool)


def flag_non_answer(texts: list[str]) -> np.ndarray:
    return np.array([bool(NON_ANSWER.match(t.lower())) for t in texts], dtype=bool)


def flag_junk(texts: list[str]) -> np.ndarray:
    """Non-linguistic strings: no word content, or a single repeated character.

    Kept separate from `non_answer` because they mean different things. A
    non-answer is an annotator telling you they could not describe the episode,
    which is information. Junk is a broken input box.
    """
    flags = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        letters = re.sub(r"[^A-Za-z]", "", text)
        if len(letters) < 2:
            flags[i] = True
        elif len(set(text)) == 1:
            flags[i] = True
    return flags


def describe_lengths(texts: list[str]) -> None:
    chars = np.array([len(t) for t in texts])
    words = np.array([len(t.split()) for t in texts])
    print(f"  {'':14}{'min':>6}{'p05':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p95':>7}{'p99':>7}{'max':>7}")
    for name, arr in (("characters", chars), ("words", words)):
        q = np.percentile(arr, [5, 25, 50, 75, 95, 99]).astype(int)
        print(
            f"  {name:<14}{arr.min():>6}{q[0]:>7}{q[1]:>7}{q[2]:>7}{q[3]:>7}"
            f"{q[4]:>7}{q[5]:>7}{arr.max():>7}"
        )
    print(f"\n  mean {chars.mean():.1f} chars, {words.mean():.1f} words")

    # A hard tool limit would pile annotations up against the cap. Show the
    # densest lengths in the upper tail so its absence is visible, not asserted.
    counts = collections.Counter(chars.tolist())
    tail = [(L, c) for L, c in counts.items() if L >= np.percentile(chars, 90)]
    tail.sort(key=lambda kv: -kv[1])
    dense = ", ".join(f"{L}ch x{c}" for L, c in tail[:8])
    print(f"  densest lengths in the top decile: {dense}")


def truncation_diagnostics(
    episodes: np.ndarray, texts: list[str], trunc: np.ndarray
) -> dict:
    """Is truncation a tool artefact? Test against length, lab, date, siblings."""
    out: dict = {}
    n_trunc = int(trunc.sum())
    chars = np.array([len(t) for t in texts])

    print(f"\n  {n_trunc} truncated annotations of {len(texts):,} ({n_trunc/len(texts):.4%})")
    if n_trunc == 0:
        return out

    lengths = sorted(chars[trunc].tolist())
    print(f"  their lengths (chars): {lengths}")
    print(f"  distinct lengths: {len(set(lengths))} of {n_trunc}")
    out["truncated_lengths"] = lengths
    # A fixed-width cut leaves nearly every truncated string the same length.
    out["length_concentration"] = len(set(lengths)) / n_trunc

    print("\n  the truncated annotations:")
    for ep, text in sorted(zip(episodes[trunc].tolist(), [texts[i] for i in np.flatnonzero(trunc)])):
        print(f"     {ep.split('+')[0]:<9} {text!r}")

    # --- lab ---------------------------------------------------------------
    labs = np.array([e.split("+")[0] for e in episodes])
    table, names = [], []
    for lab in np.unique(labs):
        m = labs == lab
        table.append([int((m & trunc).sum()), int((m & ~trunc).sum())])
        names.append(lab)
    table = np.array(table)
    print("\n  by lab:")
    for name, (bad, ok) in zip(names, table):
        rate = bad / (bad + ok)
        print(f"     {name:<10} {bad:>3} / {bad+ok:>7,}  ({rate:.4%})")
    chi2, p_lab, _, expected = stats.chi2_contingency(table)
    low = int((expected < 5).sum())
    print(f"     chi2 = {chi2:.2f}, p = {p_lab:.3f}")
    if low:
        print(f"     WARNING: {low} of {expected.size} expected counts < 5 -- chi2 is unreliable here")
    out["lab_chi2_p"] = float(p_lab)
    out["lab_low_expected_cells"] = low

    # --- date --------------------------------------------------------------
    months = np.array([e.split("+")[2][:7] for e in episodes])
    uniq = sorted(set(months.tolist()))
    mtable = np.array([[int(((months == m) & trunc).sum()),
                        int(((months == m) & ~trunc).sum())] for m in uniq])
    keep = mtable.sum(axis=1) >= 200
    chi2_m, p_month, _, exp_m = stats.chi2_contingency(mtable[keep])
    print(f"\n  by month ({int(keep.sum())} months with >=200 annotations):")
    for m, (bad, ok) in zip([u for u, k in zip(uniq, keep) if k], mtable[keep]):
        if bad:
            print(f"     {m}  {bad:>3} / {bad+ok:>6,}")
    print(f"     chi2 = {chi2_m:.2f}, p = {p_month:.3f}")
    low_m = int((exp_m < 5).sum())
    if low_m:
        print(f"     WARNING: {low_m} of {exp_m.size} expected counts < 5 -- chi2 is unreliable here")
    out["month_chi2_p"] = float(p_month)
    out["month_low_expected_cells"] = low_m

    # --- sibling length ----------------------------------------------------
    # If a tool clipped the input, the annotator's intent was longer, and the
    # co-annotators (unclipped) should look long by comparison.
    by_ep: dict[str, list[int]] = collections.defaultdict(list)
    for i, ep in enumerate(episodes.tolist()):
        by_ep[ep].append(i)
    sibling = np.full(len(texts), np.nan)
    for idxs in by_ep.values():
        if len(idxs) < 2:
            continue
        lens = chars[idxs].astype(float)
        total = lens.sum()
        for j, i in enumerate(idxs):
            sibling[i] = (total - lens[j]) / (len(idxs) - 1)
    ok = ~np.isnan(sibling)
    a, b = sibling[ok & trunc], sibling[ok & ~trunc]
    if a.size and b.size:
        u, p_sib = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(
            f"\n  mean sibling length: {a.mean():.1f} chars for truncated (n={a.size}) "
            f"vs {b.mean():.1f} for the rest (n={b.size:,})"
        )
        print(f"     Mann-Whitney U p = {p_sib:.3f}")
        out["sibling_length_p"] = float(p_sib)
        out["sibling_mean_truncated"] = float(a.mean())
        out["sibling_mean_rest"] = float(b.mean())
    return out


def alpha_deltas(
    episodes: np.ndarray, emb: np.ndarray, classes: dict[str, np.ndarray], n_boot: int
) -> dict:
    """Semantic alpha with each defect class removed, one at a time."""
    base = alpha_semantic(episodes, emb)
    point, lo, hi = bootstrap_alpha_semantic(episodes, emb, n_boot=n_boot, seed=0)
    print(f"  baseline           alpha {base.alpha:+.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
          f"   ({base.n_units:,} episodes, {base.n_pairable:,} annotations)")

    out = {"baseline": {"alpha": base.alpha, "ci": [lo, hi],
                        "n_units": int(base.n_units), "n_annotations": int(base.n_pairable)}}
    for name, flags in classes.items():
        n_drop = int(flags.sum())
        if n_drop == 0:
            print(f"  excl. {name:<13} (none found -- alpha unchanged)")
            out[name] = {"n_dropped": 0, "delta": 0.0}
            continue
        keep = ~flags
        try:
            res = alpha_semantic(episodes[keep], emb[keep])
            _, klo, khi = bootstrap_alpha_semantic(
                episodes[keep], emb[keep], n_boot=n_boot, seed=0
            )
        except ValueError as exc:
            print(f"  excl. {name:<13} not computable: {exc}")
            continue
        delta = res.alpha - base.alpha
        lost_units = base.n_units - res.n_units
        print(
            f"  excl. {name:<13} alpha {res.alpha:+.4f}  95% CI [{klo:.4f}, {khi:.4f}]"
            f"   delta {delta:+.4f}   (-{n_drop:,} annotations, -{lost_units:,} episodes)"
        )
        # Excluding a majority of the corpus does not measure the cost of a
        # defect, it measures a different corpus. Say so rather than let the
        # delta be read as comparable to the others.
        if n_drop > 0.5 * flags.size:
            print(
                f"       ^ this drops {n_drop/flags.size:.0%} of all annotations and leaves "
                f"{res.n_units:,} episodes -- not a defect class, and not comparable above"
            )
        out[name] = {
            "n_dropped": n_drop,
            "alpha": res.alpha,
            "ci": [klo, khi],
            "delta": delta,
            "episodes_lost": int(lost_units),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only use the first N episodes")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates per alpha")
    args = ap.parse_args()

    raw = fetch_annotations()
    print(f"\nloaded {len(raw):,} episodes from the annotation file")
    episodes, slots, texts = load_rows(raw, args.limit)
    print(f"  {len(texts):,} non-empty annotations across {len(set(episodes.tolist())):,} episodes")

    print("\n" + "=" * 70)
    print("LENGTH DISTRIBUTION")
    print("=" * 70)
    describe_lengths(texts)

    words = load_dictionary()
    if not words:
        print("\n  note: /usr/share/dict/words missing; truncation veto disabled")
    trunc = flag_truncated(texts, words)
    noterm = flag_no_terminal_punct(texts)
    nonans = flag_non_answer(texts)
    junk = flag_junk(texts)

    print("\n" + "=" * 70)
    print("DEFECT CLASSES")
    print("=" * 70)
    for name, flags in (("truncated mid-word", trunc), ("no terminal punctuation", noterm),
                        ("explicit non-answer", nonans), ("non-linguistic junk", junk)):
        n = int(flags.sum())
        print(f"  {name:<26} {n:>7,}  ({n/len(texts):.4%})")

    if nonans.any():
        print("\n  non-answers:")
        for text, count in collections.Counter(
            texts[i] for i in np.flatnonzero(nonans)
        ).most_common():
            print(f"     {count:>3}x {text!r}")
        # Where they sit matters more than how many there are. An episode whose
        # annotators *all* wrote "No action" contributes a perfectly-agreeing
        # unit to alpha while containing no instruction to agree about, so it
        # inflates the headline number rather than adding noise to it.
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for i, ep in enumerate(episodes.tolist()):
            grouped[ep].append(i)
        pairable = [idxs for idxs in grouped.values() if len(idxs) >= 2]
        touched = sum(1 for idxs in pairable if any(nonans[i] for i in idxs))
        whole = sum(1 for idxs in pairable if all(nonans[i] for i in idxs))
        print(
            f"     concentrated on {touched} multiply-annotated episodes, "
            f"{whole} of which are entirely non-answer"
        )
    if junk.any():
        print("\n  junk:")
        for text, count in collections.Counter(
            texts[i] for i in np.flatnonzero(junk)
        ).most_common(20):
            print(f"     {count:>3}x {text[:60]!r}")

    print("\n" + "=" * 70)
    print("IS TRUNCATION AN ANNOTATION-TOOL LIMIT?")
    print("=" * 70)
    diag = truncation_diagnostics(episodes, texts, trunc)

    print("\n" + "=" * 70)
    print("SEMANTIC ALPHA WITH EACH CLASS EXCLUDED")
    print("=" * 70)
    emb = embed(texts, tag=corpus_tag(texts))
    deltas = alpha_deltas(
        episodes,
        emb,
        {"truncated": trunc, "non-answer": nonans, "junk": junk, "no-term-punct": noterm},
        args.boot,
    )

    out = CACHE / "annotation_quality.json"
    out.write_text(
        json.dumps(
            {
                "n_annotations": len(texts),
                "n_episodes": len(set(episodes.tolist())),
                "counts": {
                    "truncated": int(trunc.sum()),
                    "no_terminal_punctuation": int(noterm.sum()),
                    "non_answer": int(nonans.sum()),
                    "junk": int(junk.sum()),
                },
                "truncation_diagnostics": diag,
                "alpha_excluding": deltas,
            },
            indent=2,
        )
    )
    print(f"\n\nresults written to {out}")


if __name__ == "__main__":
    main()
