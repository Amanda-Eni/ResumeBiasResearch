"""
Fairness intervention via sample reweighting.
The core before/after comparison for the proposal.
"""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def compute_reweighting_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """
    Up-weight samples from groups with low positive rates.
    Weight = overall_rate / group_rate, normalized.
    """
    weights = np.ones(len(y))
    overall = y.mean()
    for g in np.unique(sensitive):
        mask = sensitive == g
        group_rate = y[mask].mean()
        if group_rate > 0:
            weights[mask] = overall / group_rate
    return weights / weights.mean()


def infer_sensitive_from_names(names: np.ndarray,
                                female_names=None,
                                black_names=None) -> np.ndarray:
    """
    Infer approximate sensitive group from first name.
    Uses name lists from Bertrand & Mullainathan (2004).
    """
    if female_names is None:
        female_names = {'mary','patricia','jennifer','linda','elizabeth',
                        'barbara','susan','jessica','sarah','karen','nancy',
                        'lisa','betty','margaret','sandra','ashley','kimberly',
                        'emily','donna','michelle','carol','amanda','melissa',
                        'deborah','stephanie','dorothy','rebecca','sharon',
                        'laura','cynthia','kathleen','amy','angela','shirley',
                        'anna','brenda','pamela','emma','nicole','helen',
                        'samantha','katherine','christine','debra','rachel',
                        'carolyn','janet','catherine','maria','heather',
                        'diane','ruth','julie','olivia','joyce','virginia',
                        'victoria','kelly','lauren','christina','joan',
                        'evelyn','judith','megan','cheryl','andrea','hannah',
                        'martha','jacqueline','frances','gloria','ann',
                        'teresa','kathryn','sara','janice','jean','alice',
                        'madison','doris','abigail','julia','judy','grace',
                        'denise','amber','marilyn','beverly','danielle',
                        'theresa','sophia','marie','diana','brittany',
                        'natalie','isabella','charlotte','rose','alexis',
                        'kayla','pim','christin'}  # Pim/Christin from your sample
    if black_names is None:
        black_names = {'lakisha','keisha','tamika','latoya','shanice',
                       'tanisha','aisha','ebony','latonya','kenya',
                       'jasmine','precious','diamond','tierra','raven',
                       'iyana','jalisa','kiara','maliyah','nyla'}

    groups = []
    for n in names:
        first = str(n).strip().split()[0].lower()
        if first in black_names:
            groups.append('black')
        elif first in female_names:
            groups.append('female')
        else:
            groups.append('other')
    return np.array(groups)