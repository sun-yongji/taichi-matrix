"""
C6 Group Theory Constants for TaiChi Matrix.

Cyclic group C6 governs the hexagonal symmetry at the core of all
TaiChi Matrix modules: Router, MTP, Quant, HexAttention, Correct.
"""

# ---------------------------------------------------------------------------
# C6 Group Constants
# ---------------------------------------------------------------------------

#: Rotation angle per generator step (degrees).
C6_ROTATION_ANGLE: float = 60.0

#: Order of the cyclic group C6.
C6_ORDER: int = 6

# ---------------------------------------------------------------------------
# Derived / Tuning Constants
# ---------------------------------------------------------------------------

#: Golden-ratio-based compensation factor (|φ − 1| ≈ 0.618 → 0.0618).
GOLDEN_RATIO_COMPENSATION: float = 0.0618

#: Full reptend prime reciprocal cycle  1/7 = 0.142857 …
CYCLIC_CONSTANT_142857: int = 142857

#: Router mode boundary: below → Steady.
STEADY_THRESHOLD: float = 0.5

#: Router mode boundary: above → Perturbation / Turbulent.
PERTURBATION_THRESHOLD: float = 1.5
