#!/usr/bin/env python3
"""Minimal reproducible example of the Δₚ metric from the paper.

This script computes Jensen–Shannon divergence between two toy
next‑token probability vectors to reproduce the toy value in §5.5.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon

p = np.array([0.7, 0.2, 0.1])  # baseline token probs
q = np.array([0.6, 0.25, 0.15])  # after semantic tweak

delta_p = jensenshannon(p, q, base=2)**2  # JSD^2 is true JS divergence
print("Δₚ =", round(delta_p, 3))
