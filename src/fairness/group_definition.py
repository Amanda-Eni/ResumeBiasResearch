"""
Corrected sensitive-group definition for FairCVdb.

Background
----------
The original `infer_sensitive_from_names` used name lists from
Bertrand & Mullainathan (2004) covering only ~80 names. FairCVdb contains
5,437 unique names, so 90% of candidates fell into an "other" bucket and
one group ended up with only 10 members — making every fairness metric
meaningless.

Fix
---
FairCVdb encodes the sensitive attributes directly in the profile vector:

  * Column 0 — ethnicity: three balanced groups (~6,400 each)
  * Column 1 — gender:    two balanced groups (~9,600 each)

These columns are the ground truth. Use them for all fairness analysis.
"""
import numpy as np
from typing import Tuple


GENDER_COL = 1
ETHNICITY_COL = 0


def get_gender_groups(profiles: np.ndarray) -> np.ndarray:
    """Extract gender group labels from the profile matrix (column 1)."""
    return profiles[:, GENDER_COL].astype(int)


def get_ethnicity_groups(profiles: np.ndarray) -> np.ndarray:
    """Extract ethnicity group labels from the profile matrix (column 0)."""
    return profiles[:, ETHNICITY_COL].astype(int)


def print_group_diagnostics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            sensitive: np.ndarray,
                            attribute_name: str) -> None:
    """
    Print group sizes, selection rates, and base rates.
    Call this alongside every fairness metric so empty groups are visible.
    """
    import pandas as pd
    df = pd.DataFrame({
        'group': sensitive,
        'y_true': y_true,
        'y_pred': y_pred,
    })
    stats = df.groupby('group').agg(
        n=('y_true', 'size'),
        selection_rate=('y_pred', 'mean'),
        base_rate=('y_true', 'mean'),
    )
    stats['pct_of_total'] = (stats['n'] / len(df) * 100).round(2)
    print(f"\n--- {attribute_name} group diagnostics ---")
    print(stats)

    small_groups = stats[stats['n'] < 50]
    if len(small_groups) > 0:
        print(f"\n⚠ WARNING: {len(small_groups)} group(s) have n < 50. "
              "Fairness metrics for these groups are unreliable.")


# ---------------------------------------------------------------------------
# Backward-compatibility shim for code that still imports
# `infer_sensitive_from_names`.  Maps it to gender so notebooks that haven't
# been updated still run — but they should be migrated to the new functions.
# ---------------------------------------------------------------------------
def infer_sensitive_from_names(names: np.ndarray,
                                profiles: np.ndarray = None) -> np.ndarray:
    """DEPRECATED. Use get_gender_groups / get_ethnicity_groups instead."""
    raise RuntimeError(
        "infer_sensitive_from_names is deprecated. "
        "Use get_gender_groups(profiles) or get_ethnicity_groups(profiles)."
    )