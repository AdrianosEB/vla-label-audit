"""Turn "the labels are noisy" into "the labels cost you N points".

Measuring label noise is descriptive. The claim worth making is causal: given a
measured noise rate, how much policy performance does it destroy?

Since nobody can run the counterfactual on a real dataset -- there is no clean
version of DROID to compare against -- the move is to go the other way. Take a
dataset, inject *known* amounts of label noise, train at each level, and fit the
degradation curve. That curve converts a measured noise rate into a predicted
performance cost, which is the sentence the paper exists to write.

Three noise modes, because they are not equivalent and the literature routinely
conflates them:

* ``swap``     - the label of episode A is attached to episode B. Realistic for
                 pipeline/indexing bugs. Preserves the label distribution
                 exactly, so a model can still learn the marginal.
* ``shuffle``  - labels permuted across the whole corpus. The pure "language
                 carries no information" condition, and the right *upper bound*
                 on damage.
* ``paraphrase`` - the label is replaced by a semantically close one. Models
                 crowdsourced annotators describing the same thing differently.
                 Should be nearly harmless if the encoder is any good, which
                 makes it the control that separates real label noise from
                 mere lexical variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseResult", "inject_label_noise", "fit_degradation_curve", "predicted_cost"]


@dataclass(frozen=True)
class NoiseResult:
    """Corrupted labels plus the exact record of what was changed."""

    labels: np.ndarray
    corrupted_idx: np.ndarray
    rate: float
    mode: str

    @property
    def realised_rate(self) -> float:
        """Fraction actually changed, which is not always the requested rate.

        Under ``swap`` a pair can be drawn that already shares a label, and
        under ``paraphrase`` the nearest alternative may be the label itself.
        Report this rather than the nominal rate; the gap is small but it is the
        kind of thing that quietly biases a fitted curve.
        """
        return len(self.corrupted_idx) / len(self.labels)


def inject_label_noise(
    labels: np.ndarray,
    rate: float,
    *,
    mode: str = "swap",
    embeddings: np.ndarray | None = None,
    seed: int = 0,
) -> NoiseResult:
    """Corrupt a known fraction of labels in a controlled, reproducible way.

    Args:
        labels: ``[N]`` label ids or strings.
        rate: fraction of episodes to corrupt, in ``[0, 1]``.
        mode: ``"swap"``, ``"shuffle"``, or ``"paraphrase"``.
        embeddings: ``[N, d]`` label embeddings, required for ``"paraphrase"``.
        seed: reproducibility.
    """
    lab = np.asarray(labels).copy()
    n = lab.shape[0]
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    n_corrupt = int(round(rate * n))
    if n_corrupt == 0:
        return NoiseResult(lab, np.array([], dtype=int), rate, mode)

    if mode == "shuffle":
        idx = rng.choice(n, size=n_corrupt, replace=False)
        lab[idx] = lab[rng.permutation(idx)]
    elif mode == "swap":
        idx = rng.choice(n, size=n_corrupt - (n_corrupt % 2), replace=False)
        if idx.size == 0:
            return NoiseResult(lab, np.array([], dtype=int), rate, mode)
        a, b = idx[: idx.size // 2], idx[idx.size // 2 :]
        lab[a], lab[b] = lab[b].copy(), lab[a].copy()
    elif mode == "paraphrase":
        if embeddings is None:
            raise ValueError("paraphrase mode needs label embeddings")
        emb = np.asarray(embeddings, dtype=float)
        if emb.shape[0] != n:
            raise ValueError("embeddings and labels must align")
        norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        idx = rng.choice(n, size=n_corrupt, replace=False)
        sim = norm[idx] @ norm.T
        sim[np.arange(idx.size), idx] = -np.inf
        lab[idx] = lab[sim.argmax(axis=1)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    changed = np.flatnonzero(np.asarray(labels) != lab)
    return NoiseResult(lab, changed, rate, mode)


def fit_degradation_curve(rates: np.ndarray, scores: np.ndarray) -> dict:
    """Least-squares line through (noise rate, performance).

    Deliberately linear. With five or six noise levels and real seed variance,
    a richer functional form fits noise rather than signal, and the slope --
    "each point of label noise costs S points of success" -- is the quantity the
    paper needs anyway.

    Returns slope, intercept, r, and the two-sided p-value for the slope.
    """
    r = np.asarray(rates, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if r.shape != s.shape:
        raise ValueError("rates and scores must align")
    if r.size < 3:
        raise ValueError("need >= 3 noise levels to fit a curve")
    from scipy import stats as sps

    fit = sps.linregress(r, s)
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "p_value": float(fit.pvalue),
        "stderr": float(fit.stderr),
    }


def predicted_cost(curve: dict, measured_noise_rate: float) -> float:
    """Performance cost implied by a measured real-world noise rate.

    The final step of the argument: the audit measures how much noise exists,
    the injection experiment measures what noise costs, and this multiplies
    them. State the extrapolation assumption out loud when reporting it --
    injected noise is uniform and synthetic, real noise is neither, so this is
    an estimate of the right order of magnitude, not a point prediction.
    """
    if not 0.0 <= measured_noise_rate <= 1.0:
        raise ValueError("measured_noise_rate must lie in [0, 1]")
    return float(-curve["slope"] * measured_noise_rate)
