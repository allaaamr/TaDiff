# scripts/prepare_data_splits.py
"""
Data Splitting Script for TaDiff Training

Creates train/validation/test splits from the dataset and saves split information.
Supports cross-validation and stratified splitting based on patient characteristics.

Usage:
    python scripts/prepare_data_splits.py --data_dir ./data/lumiere --split_ratio 0.7 0.15 0.15
"""

import os
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

def get_patient_ids(csv_path: str = "data/lumiere.csv", patient_col: str = "patients") -> List[str]:
    """
    Extract patient IDs from a CSV file instead of scanning the directory.

    Args:
        csv_path: Path to the CSV file (e.g., lumiere.csv)
        patient_col: Column name that contains patient IDs

    Returns:
        List of unique patient IDs
    """
    df = pd.read_csv("data/lumiere.csv")

    # if patient_col not in df.columns:
    #     raise ValueError(f"Column '{patient_col}' not found in {csv_path}. "
    #                      f"Available columns: {df.columns.tolist()}")

    patient_ids = df["patients"].dropna().unique().tolist()
    patient_ids = sorted(patient_ids)

    print(f"Found {len(patient_ids)} unique patients in {csv_path}")
    return patient_ids

def validate_patient_data(data_dir: str, patient_id: str, required_keys: List[str]) -> bool:
    """
    Check if patient has all required data files.
    
    Args:
        data_dir: Path to data directory
        patient_id: Patient identifier
        required_keys: List of required data types (e.g., ['image', 'label', 'days', 'treatment'])
        
    Returns:
        True if all files exist, False otherwise
    """
    data_path = Path(data_dir)
    
    for key in required_keys:
        file_path = data_path / f"{patient_id}_{key}.npy"
        if not file_path.exists():
            print(f"Warning: Missing {key} file for {patient_id}")
            return False
    
    return True

def load_patient_metadata(data_dir: str, patient_id: str) -> Dict:
    """
    Load patient metadata for stratification (if available).
    
    Args:
        data_dir: Path to data directory
        patient_id: Patient identifier
        
    Returns:
        Dictionary with metadata (tumor volume, number of sessions, etc.)
    """
    try:
        # Load label to get tumor characteristics
        label_path = Path(data_dir) / f"{patient_id}_label.npy"
        labels = np.load(label_path)
        
        # Calculate basic statistics
        num_sessions = labels.shape[0]
        total_volume = np.sum(labels > 0)
        avg_volume_per_session = total_volume / num_sessions if num_sessions > 0 else 0
        
        # Load treatment info if available
        treatment_path = Path(data_dir) / f"{patient_id}_treatment.npy"
        if treatment_path.exists():
            treatments = np.load(treatment_path)
            unique_treatments = len(np.unique(treatments))
        else:
            unique_treatments = 0
        
        return {
            'patient_id': patient_id,
            'num_sessions': num_sessions,
            'total_volume': int(total_volume),
            'avg_volume': float(avg_volume_per_session),
            'num_treatments': unique_treatments
        }
    except Exception as e:
        print(f"Warning: Could not load metadata for {patient_id}: {e}")
        return {
            'patient_id': patient_id,
            'num_sessions': 0,
            'total_volume': 0,
            'avg_volume': 0,
            'num_treatments': 0
        }

def create_data_splits(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by: str = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient IDs into train/val/test sets.
    
    Args:
        patient_ids: List of all patient IDs
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        stratify_by: Optional stratification criterion ('volume', 'sessions', None)
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # First split: train+val vs test
    train_val_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_ratio,
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adjusted,
        random_state=random_seed
    )
    
    return train_ids, val_ids, test_ids

def save_splits(
    splits: Dict[str, List[str]],
    metadata: List[Dict],
    output_dir: str,
    dataset_name: str
):
    """
    Save data splits to JSON files.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' keys
        metadata: List of patient metadata dictionaries
        output_dir: Directory to save splits
        dataset_name: Name of dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits_file = output_path / f"{dataset_name}_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_file}")
    
    # Save metadata
    metadata_file = output_path / f"{dataset_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Print summary
    print("\nSplit Summary:")
    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val:   {len(splits['val'])} patients")
    print(f"  Test:  {len(splits['test'])} patients")
    print(f"  Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} patients")

def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for TaDiff')
    parser.add_argument('--data_dir', type=str, default="/l/users/alaa.mohamed/datasets/lumiere_proc",
                        help='Path to data directory')
    parser.add_argument('--dataset_name', type=str, default='lumiere',
                        help='Dataset name (sailor or lumiere)')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test split ratios (must sum to 1.0)')
    parser.add_argument('--output_dir', type=str, default='./data/splits',
                        help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--required_keys', type=str, nargs='+',
                        default=['image', 'label', 'days', 'treatment'],
                        help='Required data files for each patient')
    
    args = parser.parse_args()
    
    # Get patient IDs
    patient_ids = get_patient_ids(args.data_dir, args.dataset_name)
    
    # Validate patient data
    valid_patients = []
    for pid in patient_ids:
        if validate_patient_data(args.data_dir, pid, args.required_keys):
            valid_patients.append(pid)
    
    print(f"\nValid patients: {len(valid_patients)}/{len(patient_ids)}")
    
    if len(valid_patients) == 0:
        raise ValueError("No valid patients found!")
    
    # Load metadata
    metadata = [load_patient_metadata(args.data_dir, pid) for pid in valid_patients]
    
    # Create splits
    train_ids, val_ids, test_ids = create_data_splits(
        valid_patients,
        train_ratio=args.split_ratio[0],
        val_ratio=args.split_ratio[1],
        test_ratio=args.split_ratio[2],
        random_seed=args.seed
    )
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save splits
    save_splits(splits, metadata, args.output_dir, args.dataset_name)

if __name__ == '__main__':
    main()