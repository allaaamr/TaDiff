import os
import re
import shutil
import argparse
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------
# Helper: parse timepoint names like "week-000-1" or "week-040-2"
# ---------------------------------------------------------------------
timepoint_regex = re.compile(r"week-(\d+)(?:-(\d+))?")

def parse_timepoint(tp: str):
    """
    Returns (week_number:int, subindex:int) from a timepoint string.

    Examples
    --------
    "week-000-1" -> (0, 1)
    "week-000-2" -> (0, 2)
    "week-044"   -> (44, 0)
    """
    m = timepoint_regex.fullmatch(tp)
    if m is None:
        raise ValueError(f"Unrecognised timepoint format: {tp}")
    week = int(m.group(1))
    sub = int(m.group(2)) if m.group(2) is not None else 0
    return week, sub


# ---------------------------------------------------------------------
# Imaging + completeness handling
# ---------------------------------------------------------------------
def collect_complete_sessions(
    imaging_root: str,
    completeness_csv: str,
    output_imaging_root: str,
    log_path: str,
):
    """
    1. Reads completeness CSV.
    2. Keeps only sessions where CT1, T1, T2, FLAIR and HD-GLIO-AUTO are present.
    3. For each kept session, copies registered CT1(T1c), T1, T2, FLAIR and segmentation
       from imaging_root/.../HD-GLIO-AUTO-segmentation/registered to
       output_imaging_root/patient/timepoint.
    4. Writes a log describing dropped sessions and summary stats.

    Returns
    -------
    kept_sessions : dict[patient_id -> list of timepoints]
        Only sessions that were *actually* copied (all 5 files found).
    """
    df = pd.read_csv(completeness_csv)

    required_cols = ["CT1", "T1", "T2", "FLAIR", "HD-GLIO-AUTO"]

    log_lines = []
    weeks_dropped = 0
    patients_dropped = set()

    kept_sessions = defaultdict(list)

    for _, row in df.iterrows():
        patient = row["Patient"]
        timepoint = row["Timepoint"]

        # 1) check completeness flags in CSV
        missing = [
            col for col in required_cols
            if str(row[col]).strip().lower() != "x"
        ]

        if missing:
            weeks_dropped += 1
            patients_dropped.add(patient)
            log_lines.append(
                f"DROPPED (CSV incomplete): {patient}, {timepoint}; "
                f"missing columns: {', '.join(missing)}"
            )
            continue

        # 2) check that all expected files actually exist
        src_registered = os.path.join(
            imaging_root, patient, timepoint,
            "HD-GLIO-AUTO-segmentation", "registered"
        )
        expected_files = {
            "CT1": "CT1_r2s_bet_reg.nii.gz",   # treated as T1c
            "T1": "T1_r2s_bet_reg.nii.gz",
            "T2": "T2_r2s_bet_reg.nii.gz",
            "FLAIR": "FLAIR_r2s_bet_reg.nii.gz",
            "SEG": "segmentation.nii.gz",
        }

        missing_files = [
            key for key, fname in expected_files.items()
            if not os.path.exists(os.path.join(src_registered, fname))
        ]

        if missing_files:
            weeks_dropped += 1
            patients_dropped.add(patient)
            log_lines.append(
                f"DROPPED (files missing): {patient}, {timepoint}; "
                f"missing files: {', '.join(missing_files)}"
            )
            continue

        # 3) copy files to output structure
        dst_dir = os.path.join(output_imaging_root, patient, timepoint)
        os.makedirs(dst_dir, exist_ok=True)

        for key, fname in expected_files.items():
            src = os.path.join(src_registered, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)

        kept_sessions[patient].append(timepoint)
        log_lines.append(
            f"KEPT: {patient}, {timepoint}; copied {len(expected_files)} files."
        )

    # 4) summary
    log_lines.append("")
    log_lines.append(f"Total dropped weeks: {weeks_dropped}")
    log_lines.append(f"Total patients with at least one dropped week: "
                     f"{len(patients_dropped)}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    return kept_sessions


# ---------------------------------------------------------------------
# Clinical CSV → TaDiff-style session CSV
# ---------------------------------------------------------------------
def build_session_csv(
    clinical_csv: str,
    kept_sessions: dict,
    session_csv_out: str,
):
    """
    Builds a CSV with columns:

        patients, age, survival_months, num_ses, interval_days

    for the *kept* imaging sessions only.

    - patients: "sub-XX" where XX is derived from "Patient-0XX"
    - age: from 'Age at surgery (years)'
    - survival_months: converted from 'Survival time (weeks)'
    - num_ses: number of available timepoints for that patient
    - interval_days: differences (in days) between consecutive sessions,
      computed from timepoint names (week number * 7). Multiple sessions
      at the same week (e.g. week-000-1 and week-000-2) get 0 days
      between them.
    """
    clin = pd.read_csv(clinical_csv)

    # make Patient the index for easier lookup
    clin = clin.set_index("Patient")

    rows = []

    for patient, tps in kept_sessions.items():
        if patient not in clin.index:
            # silently skip or print a warning; here we log to console
            print(f"[WARNING] No clinical row for {patient}; skipping in session CSV.")
            continue

        # sort timepoints by actual time (and subindex)
        tps_sorted = sorted(tps, key=parse_timepoint)

        # convert timepoints to days (week_number * 7)
        days = []
        for tp in tps_sorted:
            week, _ = parse_timepoint(tp)
            days.append(week * 7)

        # compute day intervals between successive sessions
        if len(days) <= 1:
            intervals = []
        else:
            intervals = [days[i] - days[i - 1] for i in range(1, len(days))]

        num_ses = len(tps_sorted)

        # Extract clinical values
        raw_age = str(clin.loc[patient, "Age at surgery (years)"]).strip().lower()
        if raw_age in ("na", "nan", "", "none"):
            age = None
        else:
            age = float(raw_age)
        
        # ---- Safe parsing for "Survival time (weeks)" ----
        raw_surv = str(clin.loc[patient, "Survival time (weeks)"]).strip().lower()

        if raw_surv in ("na", "nan", "", "none"):
            surv_weeks = None
            survival_months = None
        else:
            surv_weeks = float(raw_surv)
            survival_months = surv_weeks / 4.34524  # weeks → months


        # convert "Patient-001" → "sub-01"
        # keep the last numeric part and cast
        pat_num = int(re.findall(r"(\d+)", patient)[-1])
        patient_id_out = f"sub-{pat_num:02d}"

        interval_str = "[" + ", ".join(str(int(d)) for d in intervals) + "]"

        rows.append(
            {
                "patients": patient_id_out,
                "age": age,
                "survival_months": survival_months,
                "num_ses": num_ses,
                "interval_days": interval_str,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(session_csv_out, index=False)
    print(f"Session CSV written to: {session_csv_out}")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare LUMIERE dataset for TaDiff."
    )
    parser.add_argument(
        "--imaging_root", type=str, default="/Users/alaa.mohamed/Downloads/imaging",
        help="Root folder with original LUMIERE imaging (patients/timepoints/...).",
    )
    parser.add_argument(
        "--completeness_csv", type=str, default="/Users/alaa.mohamed/Downloads/LUMIERE-datacompleteness.csv",
        help="CSV file describing modality completeness per patient/timepoint.",
    )
    parser.add_argument(
        "--clinical_csv", type=str, default="/Users/alaa.mohamed/Downloads/LUMIERE-Demographics_Pathology.csv",
        help="CSV with clinical data (survival, age, genomics, etc.).",
    )
    parser.add_argument(
        "--output_imaging_root", type=str, default="/Users/alaa.mohamed/Downloads/lumiere",
        help="Folder where filtered / registered images will be copied.",
    )
    parser.add_argument(
        "--session_csv_out", type=str, default="/Users/alaa.mohamed/Downloads/lumiere.csv",
        help="Path to output TaDiff-style sessions CSV.",
    )
    parser.add_argument(
        "--log_path", type=str, default="/Users/alaa.mohamed/Downloads/lumiere_preprocess_log.txt",
        help="Path to log file describing dropped sessions.",
    )

    args = parser.parse_args()

    # 1) Filter sessions + copy images
    kept_sessions = collect_complete_sessions(
        imaging_root=args.imaging_root,
        completeness_csv=args.completeness_csv,
        output_imaging_root=args.output_imaging_root,
        log_path=args.log_path,
    )

    # 2) Build TaDiff-style CSV
    build_session_csv(
        clinical_csv=args.clinical_csv,
        kept_sessions=kept_sessions,
        session_csv_out=args.session_csv_out,
    )


if __name__ == "__main__":
    main()
