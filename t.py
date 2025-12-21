import torch
import pytorch_lightning as pl

# ---- CHANGE THIS to your actual checkpoint path ----
ckpt_path = "ckpt/best.ckpt"
fixed_path = "ckpt/best_fixed.ckpt"
# -----------------------------------------------------

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# Print existing keys just for visibility
print("Checkpoint keys before fix:", ckpt.keys())

# # Add the missing version key if not present
# if "pytorch-lightning_version" not in ckpt:
ckpt["pytorch-lightning_version"] = pl.__version__
ckpt["state_dict"] = ckpt["model_state_dict"] 
model_state_dict = ckpt['state_dict']
for key, value in model_state_dict.items():
    print(f"Parameter name: {key}, Shape: {value.shape}")

#     print(f"Added pytorch-lightning_version = {pl.__version__}")
# else:
#     print("Checkpoint already has pytorch-lightning_version")

# # Save the fixed checkpoint
torch.save(ckpt, fixed_path)
print("Saved fixed checkpoint to:", fixed_path)

# #!/usr/bin/env python3
# """
# save_treatments_from_csv.py

# Usage:
#     python save_treatments_from_csv.py <csv1> <csv2> [--outdir OUTDIR]

# csv1: CSV containing columns: patient_id, num_sessions
# csv2: CSV containing columns: patient_id, num_sessions, treatment

# For each patient in csv1:
#  - find corresponding row in csv2
#  - check num_sessions match
#  - if they match, create a numpy array of shape (num_sessions,) from csv2.treatment and save to:
#     {outdir}/{Patient_ID}_treatment.npy

# treatment column supports:
#  - JSON array strings, e.g. "[0, 0, 1]"
#  - comma separated strings, e.g. "0,0,1"
#  - Python-list-like strings
#  - actual list objects if loaded by pandas
# """

# import argparse
# import os
# import sys
# import json
# import csv
# import logging
# from typing import Any, List
# import numpy as np
# import ast

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# def parse_treatment_field(val: Any) -> List[int]:
#     """
#     Parse a variety of possible representations into a Python list of ints.
#     Handles:
#       - JSON array string: "[0,1,1]"
#       - comma-separated string: "0,1,1"
#       - Python list string: "[0,1]" (ast.literal_eval)
#       - actual list object: [0,1,1]
#     Raises ValueError if cannot parse or contains non-int values.
#     """
#     if val is None:
#         raise ValueError("treatment value is None")

#     # If it's already a list-like
#     if isinstance(val, (list, tuple)):
#         arr = list(val)
#     elif isinstance(val, str):
#         s = val.strip()
#         # Try JSON
#         try:
#             parsed = json.loads(s)
#             if isinstance(parsed, (list, tuple)):
#                 arr = list(parsed)
#             else:
#                 # if parsed to not-list, fallthrough to other parsing
#                 raise ValueError
#         except Exception:
#             # Try Python literal eval (safe)
#             try:
#                 parsed = ast.literal_eval(s)
#                 if isinstance(parsed, (list, tuple)):
#                     arr = list(parsed)
#                 else:
#                     # try comma-split fallback
#                     arr = [x.strip() for x in s.split(",") if x.strip() != ""]
#             except Exception:
#                 # fallback to comma-split
#                 arr = [x.strip() for x in s.split(",") if x.strip() != ""]
#     else:
#         # try to coerce scalar -> single-element list
#         arr = [val]

#     # convert elements to ints
#     out = []
#     for x in arr:
#         if isinstance(x, (int, np.integer)):
#             out.append(int(x))
#         elif isinstance(x, (float, np.floating)):
#             # Only accept floats that are integer-valued
#             if float(x).is_integer():
#                 out.append(int(x))
#             else:
#                 raise ValueError(f"treatment contains non-integer float: {x}")
#         elif isinstance(x, str):
#             if x == "":
#                 continue
#             try:
#                 # allow numeric strings
#                 if "." in x:
#                     fx = float(x)
#                     if not fx.is_integer():
#                         raise ValueError(f"treatment contains non-integer value: {x}")
#                     out.append(int(fx))
#                 else:
#                     out.append(int(x))
#             except Exception as e:
#                 raise ValueError(f"cannot parse treatment element '{x}': {e}")
#         else:
#             raise ValueError(f"unsupported treatment element type: {type(x)} value={x}")

#     return out


# def read_csv_as_dict(csv_path: str, key_col: str) -> dict:
#     """
#     Read a CSV into a dict keyed by key_col. Keeps other columns as dict values.
#     """
#     d = {}
#     with open(csv_path, newline='') as fh:
#         reader = csv.DictReader(fh)
#         if key_col not in reader.fieldnames:
#             raise ValueError(f"Key column '{key_col}' not found in {csv_path}. Columns: {reader.fieldnames}")
#         for row in reader:
#             key = row[key_col]
#             if key in d:
#                 logging.warning(f"Duplicate key '{key}' in {csv_path} - keeping last occurrence.")
#             d[key] = row
#     return d


# def main(csv1: str, csv2: str, outdir: str):
#     os.makedirs(outdir, exist_ok=True)
#     logging.info(f"Reading CSV1: {csv1}")
#     csv1_dict = read_csv_as_dict(csv1, key_col='patients')

#     logging.info(f"Reading CSV2: {csv2}")
#     csv2_dict = read_csv_as_dict(csv2, key_col='Patient_Id')

#     total = 0
#     saved = 0
#     skipped_missing = 0
#     skipped_mismatch = 0
#     parse_errors = 0

#     for patient_id, row1 in csv1_dict.items():
#         total += 1
#         row2 = csv2_dict.get(patient_id)
#         if row2 is None:
#             logging.warning(f"[{patient_id}] not found in CSV2 -> skipping")
#             skipped_missing += 1
#             continue

#         # get num_sessions from both rows (allow string -> int)
#         try:
#             n1 = int(row1.get('num_ses', '').strip())
#         except Exception:
#             logging.error(f"[{patient_id}] invalid num_sessions in CSV1: '{row1.get('num_ses')}' -> skipping")
#             skipped_mismatch += 1
#             continue

#         try:
#             n2 = int(row2.get('num_sessions', '').strip())
#         except Exception:
#             logging.error(f"[{patient_id}] invalid num_sessions in CSV2: '{row2.get('num_sessions')}' -> skipping")
#             skipped_mismatch += 1
#             continue

#         if n1 != n2:
#             logging.warning(f"[{patient_id}] num_sessions mismatch CSV1={n1} vs CSV2={n2} -> skipping")
#             skipped_mismatch += 1
#             continue

#         # parse treatment
#         try:
#             treatment_raw = row2.get('treatment', None)
#             treatment_list = parse_treatment_field(treatment_raw)
#         except Exception as e:
#             logging.error(f"[{patient_id}] failed to parse treatment: {e} -> skipping")
#             parse_errors += 1
#             continue

#         if len(treatment_list) != n1:
#             logging.warning(f"[{patient_id}] treatment length {len(treatment_list)} != num_sessions {n1} -> skipping")
#             skipped_mismatch += 1
#             continue

#         # convert to numpy and save
#         arr = np.array(treatment_list, dtype=np.int8)
#         out_path = os.path.join(outdir, f"{patient_id}_treatment.npy")
#         try:
#             np.save(out_path, arr)
#             saved += 1
#             logging.info(f"[{patient_id}] saved {out_path} (shape={arr.shape})")
#         except Exception as e:
#             logging.error(f"[{patient_id}] failed to save numpy file: {e}")

#     logging.info("=== Summary ===")
#     logging.info(f"Total patients in CSV1: {total}")
#     logging.info(f"Saved: {saved}")
#     logging.info(f"Missing in CSV2: {skipped_missing}")
#     logging.info(f"Mismatches / length errors skipped: {skipped_mismatch}")
#     logging.info(f"Parse errors skipped: {parse_errors}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Match patients between two CSVs and save treatment arrays as .npy files.")
#     parser.add_argument("--csv1", default = "src/data/lumiere.csv", help="Path to CSV1 (patient_id, num_sessions)")
#     parser.add_argument("--csv2", default = "lumiere_csv_t.csv", help="Path to CSV2 (patient_id, num_sessions, treatment)")
#     parser.add_argument("--outdir", default="/l/users/alaa.mohamed/datasets/lumiere_proc",
#                         help="Directory to save .npy files (default: /l/users/alaa.mohamed/datasets/lumiere_proc)")
#     args = parser.parse_args()

#     try:
#         main(args.csv1, args.csv2, args.outdir)
#     except Exception as e:
#         logging.exception("Fatal error:")
#         sys.exit(2)
