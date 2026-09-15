# src/data_loader.py
"""
Load the FairCVdb dataset for resume bias research.
Dataset Structure: 24,000 synthetic profiles with blind, gender-biased, and ethnicity-biased labels.
"""
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any

def load_faircvdb(data_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Load the FairCVdb dataset from a .npy file.
    
    Args:
        data_dir: Path to the directory containing 'FairCVdb.npy'
    
    Returns:
        A dictionary with keys for train/test splits, profiles, bios, labels, etc.
    """
    data_path = Path(data_dir)
    file_path = data_path / "FairCVdb.npy"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"FairCVdb.npy not found in {data_path}. "
            "Please download it from https://github.com/BiDAlab/FairCVtest and place it in data/raw/"
        )
    
    # Load the .npy file. allow_pickle=True is required because it contains Python objects (dicts).
    # Using .item() converts the 0-d array back to the underlying dictionary[citation:17][citation:21].
    fairCV = np.load(file_path, allow_pickle=True).item()
    
    print("FairCVdb loaded successfully.")
    print("Available keys:", list(fairCV.keys()))
    
    return fairCV

if __name__ == "__main__":
    # Test the loader when running this file directly
    data = load_faircvdb()
    
    # Print shapes of key arrays to verify
    print("\n--- Train Set Shapes ---")
    print("Profiles:", data['Profiles Train'].shape)
    print("Bios:", data['Bios Train'].shape)
    print("Names:", data['Names Train'].shape)
    print("Blind Labels:", data['Blind Labels Train'].shape)
    print("Biased Labels (Gender):", data['Biased Labels Train (Gender)'].shape)
    print("Biased Labels (Ethnicity):", data['Biased Labels Train (Ethnicity)'].shape)
    
    print("\n--- Test Set Shapes ---")
    print("Profiles:", data['Profiles Test'].shape)
    print("Bios:", data['Bios Test'].shape)
    print("Names:", data['Names Test'].shape)
    print("Blind Labels:", data['Blind Labels Test'].shape)
    print("Biased Labels (Gender):", data['Biased Labels Test (Gender)'].shape)
    print("Biased Labels (Ethnicity):", data['Biased Labels Test (Ethnicity)'].shape)
    
    print("\n--- Data Sample (First Profile) ---")
    print("Profile vector length:", data['Profiles Train'][0].shape)
    print("First 5 features of profile:", data['Profiles Train'][0][:5])
    print("First name:", data['Names Train'][0])
    print("First bio (short):", data['Bios Train'][0][:200])