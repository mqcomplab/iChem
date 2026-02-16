r"""Global defaults and related utilities"""

import dataclasses

@dataclasses.dataclass(slots=True)
class BitBirchConfig:
    threshold: float = 0.30
    branching_factor: int = 1024
    merge_criterion: str = "diameter"
    refine_merge_criterion: str = "tolerance-diameter"
    refine_threshold_change: float = 0.0
    tolerance: float = 0.05
    n_features: int = 2048
    fp_kind: str = "ecfp4"


DEFAULTS = BitBirchConfig()

TSNE_SEED = 42
