"""
Generate counterfactual resume variants by swapping demographic-associated names.
This is the core experimental mechanism for your bias measurement.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Name lists from established bias research (Bertrand & Mullainathan, 2004)
# These are the canonical names used in resume audit studies
NAME_PAIRS = {
    'gender': {
        'male': ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph'],
        'female': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica']
    },
    'race': {
        'white': ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Sarah', 'Meredith', 'Carrie'],
        'black': ['Lakisha', 'Keisha', 'Tamika', 'Latoya', 'Shanice', 'Tanisha', 'Aisha', 'Ebony']
    }
}

def assign_demographic_names(df: pd.DataFrame, 
                              group_col: str = 'group',
                              name_col: str = 'name') -> pd.DataFrame:
    """
    Assign names based on demographic group labels.
    
    Args:
        df: DataFrame with a 'group' column (e.g., 'white_male', 'black_female')
        group_col: Column containing demographic group labels
        name_col: Column to create/modify with assigned names
    
    Returns:
        DataFrame with names assigned
    """
    df = df.copy()
    
    for idx, row in df.iterrows():
        group = row[group_col].lower()
        
        if 'male' in group and 'female' not in group:
            names = NAME_PAIRS['gender']['male']
        elif 'female' in group:
            names = NAME_PAIRS['gender']['female']
        elif 'white' in group:
            names = NAME_PAIRS['race']['white']
        elif 'black' in group:
            names = NAME_PAIRS['race']['black']
        else:
            names = NAME_PAIRS['gender']['male']  # default
        
        # Deterministic assignment based on index (reproducible)
        df.at[idx, name_col] = names[idx % len(names)]
    
    return df

def create_counterfactual_pairs(df: pd.DataFrame,
                                 id_col: str = 'candidate_id',
                                 name_col: str = 'name',
                                 qualification_cols: List[str] = None) -> pd.DataFrame:
    """
    Create paired counterfactual resumes: same qualifications, different names.
    
    For each candidate, create a version with a different demographic name.
    This is the core experiment design: hold qualifications constant,
    vary only the demographic signal.
    
    Returns:
        DataFrame with original + counterfactual rows, with 'pair_id' linking them.
    """
    if qualification_cols is None:
        qualification_cols = [c for c in df.columns 
                              if c not in [id_col, name_col, 'group', 'selection']]
    
    cf_rows = []
    
    for idx, row in df.iterrows():
        # Determine the "other" demographic
        original_group = row.get('group', 'unknown')
        
        if 'white' in original_group.lower():
            cf_group = original_group.replace('white', 'black').replace('White', 'Black')
        elif 'black' in original_group.lower():
            cf_group = original_group.replace('black', 'white').replace('Black', 'White')
        elif 'male' in original_group.lower() and 'female' not in original_group.lower():
            cf_group = original_group.replace('male', 'female').replace('Male', 'Female')
        elif 'female' in original_group.lower():
            cf_group = original_group.replace('female', 'male').replace('Female', 'Male')
        else:
            continue  # Skip ambiguous groups
        
        # Create counterfactual row: same qualifications, different name/group
        cf_row = row.copy()
        cf_row['group'] = cf_group
        cf_row['pair_id'] = row[id_col]  # Link to original
        cf_row['is_counterfactual'] = True
        
        # Assign a name consistent with the counterfactual group
        cf_names = _get_names_for_group(cf_group)
        cf_row[name_col] = cf_names[idx % len(cf_names)]
        
        # Qualifications remain identical
        for col in qualification_cols:
            cf_row[col] = row[col]
        
        cf_rows.append(cf_row)
    
    # Mark originals
    df = df.copy()
    df['pair_id'] = df[id_col]
    df['is_counterfactual'] = False
    
    # Combine
    result = pd.concat([df, pd.DataFrame(cf_rows)], ignore_index=True)
    
    return result


def _get_names_for_group(group: str) -> List[str]:
    """Get appropriate name list for a demographic group."""
    group = group.lower()
    if 'black' in group and 'female' in group:
        return NAME_PAIRS['race']['black']
    elif 'white' in group and 'female' in group:
        return NAME_PAIRS['race']['white']
    elif 'male' in group:
        return NAME_PAIRS['gender']['male']
    elif 'female' in group:
        return NAME_PAIRS['gender']['female']
    return NAME_PAIRS['gender']['male']