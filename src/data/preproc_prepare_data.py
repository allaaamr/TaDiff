"""
SAILOR Dataset Preprocessing Script

This script preprocesses Lumiere (brain tumor MRI) data by:
- Reading and reorienting NIfTI files
- Normalizing image intensities
- Saving processed data as numpy arrays

Author: Brian
Dataset: SAILOR MNI v2
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from monai import transforms
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Manager

# from reorient_nii import reorient

# Configure PyTorch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#  Dataset root path
lumiere_raw_path = 'data/lumiere_raw'

# MRI modality indices
# T1, T1C, FLAIR, T2 = 0, 1, 2, 3

# Segmentation label indices
BG = 0  # background
EDEMA = 1  # edema
NECROTIC = 2  # necrosis
ENHANCING = 3  # enhancing tumor

# File naming conventions
KEY_FILENAMES = {
    "mask": "segmentation.nii.gz",
    "t1": "T1_r2s_bet_reg.nii.gz",
    "t1c": "CT1_r2s_bet_reg.nii.gz",
    "flair": "Flair_r2s_bet_reg.nii.gz",
    "t2": "T2_r2s_bet_reg.nii.gz",
}

# # Patient ID lists total 27 patients in raw sailor dataset
# ALL_PATIENT_IDS = [f'sub-{i:02d}' for i in range(1, 92)]

# # # Valid patients (excluding those with missing/incorrect data)
# # VALID_PATIENT_IDS = [
# #     'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-09',
# #     'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-16', 'sub-17', 'sub-27',
# #     'sub-18', 'sub-23', 'sub-26', 'sub-05', 'sub-08', 'sub-12', 'sub-15',
# #     'sub-20', 'sub-22', 'sub-21', 'sub-25', 'sub-19',
# # ]



def read_nii(path, orientation_axcode='PLI', non_zero_norm=False, clip_percent=0.1):
    """
    Read NIfTI file and return the data array.
    
    Args:
        path (str): Path to the NIfTI file
        # orientation_axcode (str): Target orientation (e.g., 'PLI', 'RAS', 'LPI')
        non_zero_norm (bool): Whether to normalize non-zero values
        clip_percent (float): Percentile for intensity clipping (0-0.5)
    
    Returns:
        np.ndarray: Image data array
    """
    img = nib.load(path)
    # img = reorient(img, orientation_axcode)
    img_data = img.get_fdata()
    
    if non_zero_norm:
        img_data = nonzero_norm_image(img_data, clip_percent=clip_percent)
    
    return img_data


def nonzero_norm_image(image, clip_percent=0.1):
    """
    Normalize image based on non-zero values with optional intensity clipping.
    
    Args:
        image (np.ndarray): Input image
        clip_percent (float): Percentile for clipping outliers (0-0.5)
    
    Returns:
        np.ndarray: Normalized image in range [0, 1]
    """
    assert 0 <= clip_percent <= 0.5, f"clip_percent must be in [0, 0.5], got {clip_percent}"
    
    nz_mask = image > 0
    
    if image[nz_mask].size == 0:
        print(f"Warning: Image has no non-zero values. Shape: {image.shape}, "
              f"Min: {image.min():.4f}, Max: {image.max():.4f}")
        return image
    
    # Clip outliers if specified
    if clip_percent > 0:
        minval = np.percentile(image[nz_mask], clip_percent)
        maxval = np.percentile(image[nz_mask], 100 - clip_percent)
        image[nz_mask & (image < minval)] = minval
        image[nz_mask & (image > maxval)] = maxval
    
    # Z-score normalization
    y = image[nz_mask]
    image_mean = np.mean(y)
    image_std = np.std(y)
    
    assert image_std != 0.0, f"Image std is zero: {image_std}"
    
    image = (image - image_mean) / image_std
    
    # Scale to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image


def get_session_list(file_csv="data/lumiere.csv"):
    """
    Load time sessions and intervals  information from CSV file.
    
    Args:
        file_csv (str): Name of the CSV file containing patient information
    
    Returns:
        tuple: (times_list, treatment_list) dictionaries keyed by patient ID
            - times_list: Cumulative days from baseline for each session
            - treatment_list: Treatment type (0: CRT, 1: TMZ, 2: None, 3: Unknown)
    """
    times_list = {}
    treatment_list = {}
    
    df = pd.read_csv(file_csv)
    # df.set_index("patients", inplace=True)
    
    for _, row in df.iterrows():
        patient_id = row["patients"]
        
        # Parse interval_days string to array
        inv_times = row["interval_days"]  # e.g. "[7, 14, 21, ...]"
        time_inv = np.array([int(s) for s in inv_times.strip("[]").split(",") if s.strip() != ""])


        if len(time_inv) == 0 or time_inv[0] != 0:
            times = np.insert(time_inv, 0, 0)
        else:
            times = time_inv
        
        # Treatment encoding: 0=CRT (sessions 0-3), 1=TMZ (sessions 4+)
        treatment_list[patient_id] = np.array(
            [1 if i > 3 else 0 for i in range(len(time_inv) + 1)]
        )
    
    return times_list, treatment_list

import os
import pandas as pd

def get_file_dict(root=lumiere_raw_path, csv_path="data/lumiere.csv"):
    """
    Create a nested dictionary of file paths for only the patients listed in the CSV.
    
    Returns:
        dict: 
            {patient_id: {session_id: {modality: filepath}}}
    """
    file_dict = {}

    # --- Load patient IDs from CSV ---
    df = pd.read_csv(csv_path)
    csv_patient_ids = sorted(df["patients"].unique())

    # --- Get only folders that match the CSV patient IDs ---
    patient_ids = [
        f.name
        for f in os.scandir(root)
        if f.is_dir() and f.name in csv_patient_ids
    ]

    for patient_id in patient_ids:
        sessions = {}
        patient_path = os.path.join(root, patient_id)

        # Get session directories inside patient folder
        session_ids = sorted([
            f.name
            for f in os.scandir(patient_path)
            if f.is_dir()
        ])

        # Build file paths for each session
        for session_id in session_ids:
            session_path = os.path.join(root, patient_id, session_id)

            files = {
                key: os.path.join(session_path, filename)
                for key, filename in KEY_FILENAMES.items()
            }

            sessions[session_id] = files

        file_dict[patient_id] = sessions

    return file_dict


# def get_file_dict(root=lumiere_raw_path):
#     """
#     Create a nested dictionary of file paths for all patients and sessions.
    
    
#     Returns:
#         dict: Nested dict structure:
#             {patient_id: {session_id: {modality: filepath}}}
#     """
#     file_dict = {}
#     patient_ids = sorted([
#         f.name for f in os.scandir(root)
#         if f.is_dir()
#     ])



#     for patient_id in patient_ids and not "Patient-001":
#         sessions = {}
#         patient_path = os.path.join(root, patient_id)
        
#         # Get all session directories
#         session_ids = sorted([
#             os.path.basename(f.path)
#             for f in os.scandir(patient_path)
#             if f.is_dir()
#         ])
        
#         # Build file paths for each session
#         for session_id in session_ids:
#             files = {
#                 key: os.path.join(root, patient_id, session_id, filename)
#                 for key, filename in KEY_FILENAMES.items()
#             }
#             sessions[session_id] = files
        
#         file_dict[patient_id] = sessions
    
#     return file_dict


def save_session_data(file_dict=None, save_path='./lumiere', root=lumiere_raw_path):
    """
    Process and save all patient data as numpy arrays.
    
    For each patient, saves:
        - {patient_id}_image.npy: (M*T, H, W, D) array of MRI modalities
        - {patient_id}_label.npy: (T, H, W, D) array of segmentation masks
        - {patient_id}_days.npy: (T,) array of cumulative days
        - {patient_id}_treatment.npy: (T,) array of treatment types
    
    Where M=4 modalities, T=number of timepoints, H,W,D=spatial dimensions
    
    Args:
        file_dict (dict): File path dictionary from get_file_dict()
        save_path (str): Output directory for numpy files
    """
    if file_dict is None:
        file_dict = get_file_dict()
    
    os.makedirs(save_path, exist_ok=True)
    
    session_list, treatment_list = get_session_list()
    patient_ids = list(file_dict.keys())
    print(patient_ids)
    
    for patient_id in patient_ids:
        print(f'Processing {patient_id}...')
        
        session_ids = sorted(list(file_dict[patient_id].keys()))
        
        images = []
        labels = []
        
        # Process each modality and session
        modalities = ['t1', 't1c', 'flair', 't2']
        for i, modality in enumerate(modalities):
            for session_id in session_ids:
                # Load and normalize image
                img_path = file_dict[patient_id][session_id][modality]
                img = read_nii(img_path, non_zero_norm=True, clip_percent=0.2)
                images.append(img)
                
                # Load and merge segmentation masks (only once per session)
                if i == 0:
                    merged_labels = np.zeros_like(img)
                    
                    # Edema mask
                    mask_path = file_dict[patient_id][session_id]['mask']
                    mask = read_nii(mask_path)
                    merged_labels[mask > 0] = ENHANCING
                    

                    
                    labels.append(merged_labels.astype(np.int8))
        
        # Stack and save
        image = np.stack(images, axis=0).astype(np.float32)
        label = np.stack(labels, axis=0).astype(np.int8)
        
        # Calculate cumulative days
        day_interval = session_list[patient_id]
        days = np.array([sum(day_interval[:i+1]) for i in range(len(day_interval))])
        
        treatment = treatment_list[patient_id]
        
        # Save to disk
        np.save(os.path.join(save_path, f"{patient_id}_image.npy"), image)
        np.save(os.path.join(save_path, f"{patient_id}_label.npy"), label)
        np.save(os.path.join(save_path, f"{patient_id}_days.npy"), days)
        np.save(os.path.join(save_path, f"{patient_id}_treatment.npy"), treatment)
        
        print(f"  Saved: Image {image.shape}, Label {label.shape}, "
              f"Days {days.shape}, Treatment {treatment.shape}")
    
    print(f"\nPreprocessing complete! Data saved to: {save_path}")


if __name__ == "__main__":
    """Sanity check and run preprocessing"""
    
    # # Verify file structure
    # # file_dict = get_file_dict(patient_ids=VALID_PATIENT_IDS, root=ROOT)
    # file_dict = get_file_dict(patient_ids=['sub-17'], root=sailor_raw_path)
    # print(f'Total patients: {len(file_dict)}')
    # print(f'Patient IDs: {list(file_dict.keys())}')
    # print(f"\nExample - sub-17 sessions: {list(file_dict['sub-17'].keys())}")
    # print(f"Example - sub-17/ses-01 files: {list(file_dict['sub-17']['ses-01'].keys())}")
    
    # Run preprocessing
    print("\nStarting preprocessing...\n")
    save_session_data()
