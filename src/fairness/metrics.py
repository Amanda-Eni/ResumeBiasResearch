"""
Fairness metrics for resume screening bias evaluation.
Implements: Demographic Parity, Equal Opportunity, Disparate Impact.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def demographic_parity_difference(y_pred: np.ndarray, 
                                   sensitive: np.ndarray) -> float:
    """
    Demographic Parity Difference: max(P(Ŷ=1|group=A)) - min(P(Ŷ=1|group=B))
    
    Measures whether positive outcomes are distributed equally across groups.
    A value of 0 indicates perfect parity; higher values indicate disparity.
    """
    groups = np.unique(sensitive)
    rates = {}
    
    for g in groups:
        mask = sensitive == g
        rates[g] = y_pred[mask].mean()
    
    return max(rates.values()) - min(rates.values())


def equal_opportunity_difference(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  sensitive: np.ndarray) -> float:
    """
    Equal Opportunity Difference: max(TPR_A) - min(TPR_B)
    
    TPR = P(Ŷ=1 | Y=1, group). Measures whether qualified candidates
    from different groups have equal chances of being selected.
    """
    groups = np.unique(sensitive)
    tprs = {}
    
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() > 0:
            tprs[g] = y_pred[mask].mean()
        else:
            tprs[g] = 0.0
    
    return max(tprs.values()) - min(tprs.values())


def disparate_impact_ratio(y_pred: np.ndarray,
                           sensitive: np.ndarray,
                           reference_group: str = None) -> Dict[str, float]:
    """
    Disparate Impact Ratio (Four-Fifths Rule).
    
    DIR = P(Ŷ=1|group) / P(Ŷ=1|reference)
    
    Under EEOC Uniform Guidelines (29 C.F.R. 1607), a ratio below 0.80
    is considered evidence of adverse impact.
    """
    groups = np.unique(sensitive)
    
    if reference_group is None:
        # Use group with highest selection rate as reference
        rates = {g: y_pred[sensitive == g].mean() for g in groups}
        reference_group = max(rates, key=rates.get)
        print(f"Using '{reference_group}' as reference (highest selection rate)")
    
    ref_rate = y_pred[sensitive == reference_group].mean()
    
    ratios = {}
    for g in groups:
        if g == reference_group:
            ratios[g] = 1.0
        else:
            group_rate = y_pred[sensitive == g].mean()
            ratios[g] = group_rate / ref_rate if ref_rate > 0 else 0.0
    
    return ratios


def compute_all_fairness_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  sensitive: np.ndarray) -> Dict:
    """
    Compute all fairness metrics in one call.
    """
    return {
        'demographic_parity_difference': demographic_parity_difference(y_pred, sensitive),
        'equal_opportunity_difference': equal_opportunity_difference(y_true, y_pred, sensitive),
        'disparate_impact_ratios': disparate_impact_ratio(y_pred, sensitive)
    }

# Alias for notebook compatibility
compute_all = compute_all_fairness_metrics