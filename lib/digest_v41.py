"""Data digests for the revision-41 figures: sha256 of the arrays handed to the plotting calls."""
import hashlib

import numpy as np


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
