"""
MIUGlioma Dataset Preprocessing Script

Given a dataset folder structure like:
root/
  PatientID_0003/
    Timepoint_1/
      PatientID_0003_Timepoint_1_brain_t1n.nii.gz
      PatientID_0003_Timepoint_1_brain_t1c.nii.gz
      PatientID_0003_Timepoint_1_brain_t2f.nii.gz   (FLAIR)
      PatientID_0003_Timepoint_1_brain_t2w.nii.gz   (T2)
      PatientID_0003_Timepoint_1_tumorMask.nii.gz
    Timepoint_2/
      ...

And a CSV with columns including:
patients, days_interval, treatment, and multiple genomic columns

This script:
1) Scans each patient folder and each Timepoint_* session folder
2) Loads modalities in order: T1, T1C, FLAIR, T2
3) Saves:
   - {patient_id}_image.npy   shape: (S, 4, H, W, D)
   - {patient_id}_label.npy   shape: (S, H, W, D)
   - {patient_id}_days.npy    shape: (S,) cumulative days from days_interval
   - {patient_id}_treatment.npy shape: (S,) encoded treatments CRT=0, TMZ=1, IMT=2
   - {patient_id}_geno.npy    shape: (G,) fixed-order genomic vector
"""

import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


# ---- Modality order required by you ----
MODALITIES_ORDER = ["t1n", "t1c", "t2f", "t2w"]  # T1, T1C, FLAIR, T2
MODALITY_TO_INDEX = {m: i for i, m in enumerate(MODALITIES_ORDER)}

# tumor mask filename token
MASK_TOKEN = "tumorMask"

# Treatment encoding requested
TREATMENT_MAP = {"CRT": 0, "TMZ": 1, "IMT": 2}

# Genomic columns (use those that actually exist in the CSV; saved in this fixed order)
GENO_COLUMNS_ORDER = [
    "IDH1 mutation",
    "IDH2 mutation",
    "1p/19q",
    "ATRX mutation",
    "MGMT methylation",
    "BRAF V600E mutation",
    "TERT promoter mutation",
    "Chromosome 7 gain and Chromosome 10 loss",
    "H3-3A mutation",
    "EGFR amplification",
    "PTEN mutation",
    "CDKN2A/B deletion",
    "TP53 alteration",
]


def read_nii(path: str) -> np.ndarray:
    """Load NIfTI file and return data as float32 (or int16 for masks if you prefer)."""
    img = nib.load(path)
    data = img.get_fdata()
    return data


def nonzero_zscore_to_01(x: np.ndarray, clip_percent: float = 0.2) -> np.ndarray:
    """Optional: normalize non-zero voxels then scale to [0,1]."""
    x = x.astype(np.float32, copy=False)
    nz = x > 0
    if not np.any(nz):
        return x

    if clip_percent and 0.0 < clip_percent < 0.5:
        lo = np.percentile(x[nz], clip_percent)
        hi = np.percentile(x[nz], 100.0 - clip_percent)
        x[nz & (x < lo)] = lo
        x[nz & (x > hi)] = hi

    vals = x[nz]
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma > 0:
        x = (x - mu) / sigma

    mn = float(x.min())
    mx = float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    return x


def parse_listlike(value, cast=float):
    """
    Parse list-like cell values such as:
      "[0, 30, 60]" or "0,30,60" or "0 30 60"
    Returns a python list (possibly empty).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return [cast(v) for v in value]

    s = str(value).strip()
    if s == "":
        return []

    # Try Python literal (e.g., "[1,2,3]")
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [cast(v) for v in obj]
    except Exception:
        pass

    # Fallback: split on commas/whitespace
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p != ""]
    out = []
    for p in parts:
        try:
            out.append(cast(p))
        except Exception:
            # ignore non-parsable tokens
            continue
    return out


def encode_treatments(treatment_seq):
    """
    Encode session-wise treatments using CRT->0, TMZ->1, IMT->2.
    Unknown/unseen values -> -1.
    """
    enc = []
    for t in treatment_seq:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            enc.append(-1)
            continue
        key = str(t).strip().upper()
        enc.append(TREATMENT_MAP.get(key, -1))
    return np.array(enc, dtype=np.int64)


def encode_geno_value(v):
    """
    Robust encoding for genomic cells.
    - If numeric already -> float(v)
    - If strings like yes/no/true/false/pos/neg/present/absent -> 1/0
    - Otherwise: try to parse to float; if fails -> np.nan
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return float(v)
    if isinstance(v, (float, np.floating)):
        return float(v)

    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "pos", "positive", "present", "mut", "mutant", "methylated", "gain"}:
        return 1.0
    if s in {"0", "false", "f", "no", "n", "neg", "negative", "absent", "wt", "wildtype", "unmethylated", "loss"}:
        return 0.0

    # common unknowns
    if s in {"na", "n/a", "nan", "unknown", "unk", ""}:
        return np.nan

    # try numeric parse
    try:
        return float(s)
    except Exception:
        return np.nan


def find_sessions(patient_dir: str):
    """Return sorted list of session folder names like Timepoint_1, Timepoint_2, ..."""
    sessions = []
    for entry in os.scandir(patient_dir):
        if entry.is_dir() and entry.name.lower().startswith("timepoint"):
            sessions.append(entry.name)
    # natural sort by the numeric suffix if present
    def key_fn(name):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 10**9
    return sorted(sessions, key=key_fn)


def index_session_files(session_dir: str):
    """
    Return dict:
      {
        "t1n": path,
        "t1c": path,
        "t2f": path,
        "t2w": path,
        "mask": path
      }
    based on filename tokens.
    """
    found = {}
    for entry in os.scandir(session_dir):
        if not entry.is_file():
            continue
        fn = entry.name

        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
            low = fn.lower()

            # mask
            if MASK_TOKEN.lower() in low:
                found["mask"] = entry.path
                continue

            # modalities (look for _brain_<token> or just <token> anywhere)
            for mod in MODALITIES_ORDER:
                if re.search(rf"(^|[_\-]){re.escape(mod)}([_\-\.]|$)", low) or f"brain_{mod}" in low:
                    found[mod] = entry.path
                    break
    return found


def preprocess_patient(patient_id: str, patient_dir: str, csv_row: pd.Series, out_dir: str,
                       normalize: bool = False, clip_percent: float = 0.2):
    sessions = find_sessions(patient_dir)
    if len(sessions) == 0:
        print(f"[WARN] No sessions found for {patient_id} in {patient_dir}")
        return

    # ---- Load images/masks ----
    image_sessions = []
    mask_sessions = []

    expected_shape = None

    for sess in sessions:
        sess_dir = os.path.join(patient_dir, sess)
        files = index_session_files(sess_dir)

        # Check all modalities exist
        missing = [m for m in MODALITIES_ORDER if m not in files]
        if "mask" not in files:
            missing.append("mask")
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} for patient {patient_id} session {sess} in {sess_dir}"
            )

        # Load modalities in required order -> (4, H, W, D)
        mods_data = []
        for m in MODALITIES_ORDER:
            arr = read_nii(files[m]).astype(np.float32)
            if normalize:
                arr = nonzero_zscore_to_01(arr, clip_percent=clip_percent)
            mods_data.append(arr)

        mods_data = np.stack(mods_data, axis=0)  # (4, H, W, D)

        # Load mask -> (H, W, D)
        mask = read_nii(files["mask"])
        # keep as int (0/1/2/...) if stored that way
        mask = np.asarray(mask).astype(np.int16)

        if expected_shape is None:
            expected_shape = mods_data.shape[1:]  # HWD
        else:
            if tuple(mods_data.shape[1:]) != tuple(expected_shape) or tuple(mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Shape mismatch for {patient_id} {sess}. "
                    f"Expected {expected_shape}, got mods {mods_data.shape[1:]} mask {mask.shape}. "
                    f"Resampling not implemented in this script."
                )

        image_sessions.append(mods_data)
        mask_sessions.append(mask)

    images_sc = np.stack(image_sessions, axis=0).astype(np.float32)  # (S, 4, H, W, D)
    labels_sc = np.stack(mask_sessions, axis=0).astype(np.int16)     # (S, H, W, D)

    S, C, H, W, D = images_sc.shape
    images = images_sc.reshape(S * C, D, H, W)
    S, H, W, D = labels_sc.shape
    labels = labels_sc.reshape(S, D, H, W)         

    # ---- Days (cumulative) from CSV: derive from days_interval ----
    # Your CSV columns include both "days" and "days_interval"; you requested derive from days_interval.
    day_intervals = parse_listlike(csv_row.get("days_interval", ""), cast=float)
    day_intervals = [float(x) for x in day_intervals]

    # If days_interval length doesn’t match number of sessions, we still compute based on what’s available
    # and then crop/pad to S.
    cum_days = np.cumsum(np.array(day_intervals, dtype=np.float32)) if len(day_intervals) > 0 else np.array([], dtype=np.float32)

    S = images.shape[0]
    if len(cum_days) < S:
        # pad by repeating last value (or zeros if empty)
        pad_val = float(cum_days[-1]) if len(cum_days) > 0 else 0.0
        cum_days = np.concatenate([cum_days, np.full((S - len(cum_days),), pad_val, dtype=np.float32)], axis=0)
    elif len(cum_days) > S:
        cum_days = cum_days[:S]

    # ---- Treatments (session-wise) from CSV ----
    treatment_seq = parse_listlike(csv_row.get("treatment", ""), cast=str)
    treatment_enc = encode_treatments(treatment_seq)

    if len(treatment_enc) < S:
        pad_val = int(treatment_enc[-1]) if len(treatment_enc) > 0 else -1
        treatment_enc = np.concatenate([treatment_enc, np.full((S - len(treatment_enc),), pad_val, dtype=np.int64)], axis=0)
    elif len(treatment_enc) > S:
        treatment_enc = treatment_enc[:S]

    # ---- Genomics vector (patient-level) ----
    # Use whichever of GENO_COLUMNS_ORDER exist in the CSV.
    geno_cols = [c for c in GENO_COLUMNS_ORDER if c in csv_row.index]
    geno_vec = np.array([encode_geno_value(csv_row.get(c, np.nan)) for c in geno_cols], dtype=np.float32)

    # ---- Save ----
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{patient_id}_image.npy"), images)
    np.save(os.path.join(out_dir, f"{patient_id}_label.npy"), labels)
    np.save(os.path.join(out_dir, f"{patient_id}_days.npy"), cum_days.astype(np.float32))
    np.save(os.path.join(out_dir, f"{patient_id}_treatment.npy"), treatment_enc.astype(np.int64))
    np.save(os.path.join(out_dir, f"{patient_id}_geno.npy"), geno_vec.astype(np.float32))

    print(
        f"[OK] {patient_id}: image {images.shape}, label {labels.shape}, "
        f"days {cum_days.shape}, treatment {treatment_enc.shape}, geno {geno_vec.shape} (cols={len(geno_cols)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="MMContraFuse/data/MIUGlioma/MU-Glioma-Raw",
                    help="Path to the directory containing patient folders (e.g., .../MU-Glioma-Raw)")
    ap.add_argument("--csv_path", type=str, default="TaDiff/data/MUGlioma.csv", help="Path to the CSV file")
    ap.add_argument("--out_dir", type=str, default="TaDiff/data/miu", help="Where to save the .npy outputs")
    ap.add_argument("--normalize", action="store_true", help="Apply non-zero zscore->minmax normalization to images")
    ap.add_argument("--clip_percent", type=float, default=0.2, help="Clipping percentile for normalization (0-0.5)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    if "patients" not in df.columns:
        raise KeyError("CSV must contain a 'Patient_ID' column.")

    # Build quick lookup by patient id (string)
    df["patients"] = df["patients"].astype(str)
    csv_lookup = df.set_index("patients", drop=False)

    # Iterate patient folders on disk
    patient_folders = sorted([d.name for d in os.scandir(args.root_dir) if d.is_dir()])

    for patient_id in patient_folders:
        if patient_id not in csv_lookup.index:
            print(f"[WARN] Patient folder '{patient_id}' not found in CSV. Skipping.")
            continue

        patient_dir = os.path.join(args.root_dir, patient_id)
        row = csv_lookup.loc[patient_id]

        preprocess_patient(
            patient_id=patient_id,
            patient_dir=patient_dir,
            csv_row=row,
            out_dir=args.out_dir,
            normalize=args.normalize,
            clip_percent=args.clip_percent,
        )


if __name__ == "__main__":
    print("Hi")
    main()
    print("Hi")
