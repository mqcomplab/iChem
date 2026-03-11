r"""Misc. utility functions"""

import numpy as np

__all__ = [
    "min_safe_uint"
]


def min_safe_uint(nmax: int) -> np.dtype:
    r"""Returns the min uint dtype that holds a (positive) py int, excluding "object".

    Input must be a positive python integer.
    """
    out = np.min_scalar_type(nmax)
    # Check if the dtype is a pointer to a python bigint
    if out.hasobject:
        raise ValueError(f"n_samples: {nmax} is too large to hold in a uint64 array")
    return out
