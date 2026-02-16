"""
Constants and default feature definitions for the active learning pipeline.
"""

# RACs that are invariant in the design space and should be excluded from models
INVARIANT_FEATURES = [
    "lc-I-0-ax",
    "lc-I-0-eq",
    "D_lc-chi-0-ax",
    "D_lc-Z-0-ax",
    "D_lc-I-0-ax",
    "D_lc-I-1-ax",
    "D_lc-I-2-ax",
    "D_lc-I-3-ax",
    "D_lc-I-4-ax",
    "D_lc-T-0-ax",
    "D_lc-S-0-ax",
    "D_lc-chi-0-eq",
    "D_lc-Z-0-eq",
    "D_lc-I-0-eq",
    "D_lc-I-1-eq",
    "D_lc-I-2-eq",
    "D_lc-I-3-eq",
    "D_lc-I-4-eq",
    "D_lc-T-0-eq",
    "D_lc-S-0-eq",
]

# Core features always included (before RACs)
CORE_FEATURE_NAMES = [
    "OHE_mn",
    "OHE_fe",
    "ox",
    "spin",
    "monocharge",
    "tetracharge",
]

# Target column names in labeled data
TARGET_HAT = "HAT (kcal/mol)"
TARGET_REBOUND = "rebound (kcal/mol)"

# Default EHVI / selection
DEFAULT_EHVI_TOP_K = 100_000
DEFAULT_N_CLUSTERS = 200
DEFAULT_BATCH_SIZE = 10_000

# Tanimoto screening (optional before EHVI)
DEFAULT_TANIMOTO_THRESHOLD = 0.842
