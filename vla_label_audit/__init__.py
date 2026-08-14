"""vla-label-audit: measuring whether robot-learning datasets say what they think they say."""

from .agreement import (
    AgreementResult,
    bootstrap_alpha_ci,
    cosine_distance_matrix,
    exact_match_distance_matrix,
    fleiss_kappa,
    krippendorff_alpha,
    per_unit_disagreement,
)
from .crossmodal import (
    AlignmentResult,
    cca_alignment,
    effective_rank,
    gaussian_mi_from_cca,
    instruction_space_report,
    knn_indices,
    mutual_information_ksg,
    neighborhood_disagreement,
    neighborhood_overlap,
    rank_correlation_across_views,
)
from .scalable import alpha_nominal, alpha_semantic, bootstrap_alpha_semantic
from .noise import NoiseResult, fit_degradation_curve, inject_label_noise, predicted_cost

__version__ = "0.1.0"
