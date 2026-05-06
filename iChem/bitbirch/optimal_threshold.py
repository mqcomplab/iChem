from bblean.similarity import estimate_jt_std, jt_isim_packed
from bblean.fingerprints import pack_fingerprints
import numpy as np # type: ignore

def optimal_threshold(fps: np.ndarray,
                      packed: bool = True,
                      factor: float = 3.5,):
    """Estimate an optimal threshold for clustering based on iSIM and iSIM-sigma."""

    if not packed:
        fps = pack_fingerprints(fps)
    
    isim_sigma = estimate_jt_std(fps)
    isim = jt_isim_packed(fps)

    return isim + factor * isim_sigma
