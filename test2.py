import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# # ----------------------------------------------------------------------
# # CONFIG
# # ----------------------------------------------------------------------
# IMG_ROOT = "/l/users/alaa.mohamed/datasets/lumiere_proc"
# CSV_PATH = "src/data/lumiere.csv"

# # Column names in the CSV:
# # patients, age, survival_months, num_ses, interval_days
# # interval_days is assumed to be something like a list/array-like string
# # e.g. "[0, 30, 60]" or "0,30,60". Adjust parsing below if needed.

# def parse_interval_days(value):
#     """
#     Convert the `interval_days` column text into a numpy array of ints.
#     Adjust this if your format is different.
#     """
#     if isinstance(value, (list, np.ndarray)):
#         return np.array(value, dtype=int)

#     # Convert string to list of ints, handling a few common formats
#     s = str(value).strip()
#     # Remove brackets if present
#     s = s.strip("[]()")
#     if not s:
#         return np.array([], dtype=int)
#     # Split on comma or whitespace
#     if "," in s:
#         parts = [p.strip() for p in s.split(",") if p.strip()]
#     else:
#         parts = [p.strip() for p in s.split() if p.strip()]
#     return np.array([int(p) for p in parts], dtype=int)


# def format_interval_days(arr: np.ndarray) -> str:
#     """Convert numpy array back to a string to store in CSV."""
#     return "[" + ", ".join(str(int(x)) for x in arr) + "]"


# def process_patient(row):
#     patient_id = row["patients"]
#     print(f"\n{'='*80}")
#     print(f"Processing patient: {patient_id}")
#     print(f"{'='*80}")

#     # Build file paths
#     img_path   = os.path.join(IMG_ROOT, f"{patient_id}_image.npy")
#     lbl_path   = os.path.join(IMG_ROOT, f"{patient_id}_label.npy")
#     days_path  = os.path.join(IMG_ROOT, f"{patient_id}_days.npy")
#     treat_path = os.path.join(IMG_ROOT, f"{patient_id}_treatment.npy")

#     # Check existence
#     for p in [img_path, lbl_path, days_path, treat_path]:
#         if not os.path.exists(p):
#             print(f"  [WARNING] Missing file for patient {patient_id}: {p}")
#             print("  Skipping this patient.")
#             return row  # unchanged

#     # Load arrays
#     img   = np.load(img_path)   # shape: (C_full * S, H, W, D)
#     lbl   = np.load(lbl_path)   # shape: (S, H, W, D)
#     days  = np.load(days_path)  # shape: (S,)
#     treat = np.load(treat_path) # shape: (S,)

#     # Parse interval_days from CSV
#     interval_days = parse_interval_days(row["interval_days"])

#     # Basic consistency checks (you can make this stricter if you want)
#     S_csv = int(row["num_ses"])
#     S_lbl = lbl.shape[0]
#     S_days_npy = days.shape[0]
#     S_treat = treat.shape[0]
#     S_interval = interval_days.shape[0]
#     C_times_S = img.shape[0]

#     print(f"  num_ses (CSV): {S_csv}")
#     print(f"  Sessions from label.npy     : {S_lbl}")
#     print(f"  Sessions from days.npy      : {S_days_npy}")
#     print(f"  Sessions from treatment.npy : {S_treat}")
#     print(f"  Sessions from interval_days : {S_interval}")
#     print(f"  image.npy first dim (C*S)   : {C_times_S}")

#     # If there are no interval days, nothing to do
#     if S_interval == 0:
#         print("  [INFO] interval_days empty, nothing to drop.")
#         return row

#     zero_int_idx = np.where(interval_days == 0)[0]  # indices in [0..S-2]
#     if zero_int_idx.size == 0:
#         print("  [INFO] No zero intervals, nothing to drop.")
#         return row

#     print(f"  Zero intervals at indices: {zero_int_idx.tolist()}")

#     # Before-shapes (for printing)
#     print("\n  SHAPES BEFORE DELETION:")
#     print(f"    image      : {img.shape}")
#     print(f"    label      : {lbl.shape}")
#     print(f"    days       : {days.shape}")
#     print(f"    treatment  : {treat.shape}")
#     print(f"    intervals  : {interval_days.shape}  (values={interval_days.tolist()})")

#     # --- LABEL: delete corresponding sessions (i+1) ---
#     sessions_to_remove = zero_int_idx + 1      # interval i → session i+1
#     sessions_to_remove = np.unique(sessions_to_remove)
#     num_removed = int(sessions_to_remove.size)
#     lbl_new = np.delete(lbl, sessions_to_remove, axis=0)

#     # --- IMAGE: delete all 4 modalities for those sessions ---
#     img_delete_idx = []
#     for idx in zero_int_idx:
#         start = 4 * (idx + 1)     # (4 + 4*idx)
#         end   = start + 4         # up to (4 + 4*idx + 3)
#         img_delete_idx.extend(range(start, end))

#     img_delete_idx = np.unique(img_delete_idx)
#     img_new = np.delete(img, img_delete_idx, axis=0)

#     # --- DAYS / TREATMENT / INTERVALS: delete at interval indices themselves ---

#     intervals_new = np.delete(interval_days, zero_int_idx, axis=0)

#     days_new = days
#     treat_new = treat
#     if zero_int_idx.size >= 1: 
#         if  zero_int_idx[0]==0:
#             zero_int_idx = np.delete(zero_int_idx, 0)

#         if zero_int_idx.size > 0 :
#             days_new      = np.delete(days,       zero_int_idx, axis=0)
#             treat_new     = np.delete(treat,      zero_int_idx, axis=0)

#     # Print shapes after deletion
#     print("\n  SHAPES AFTER DELETION:")
#     print(f"    image      : {img_new.shape}")
#     print(f"    label      : {lbl_new.shape}")
#     print(f"    days       : {days_new.shape}")
#     print(f"    treatment  : {treat_new.shape}")
#     print(f"    intervals  : {intervals_new.shape}  (values={intervals_new.tolist()})")

#     # --- Save back to disk ---
#     np.save(img_path,   img_new)
#     np.save(lbl_path,   lbl_new)
#     np.save(days_path,  days_new)
#     np.save(treat_path, treat_new)

#     # --- Update CSV row ---
#     new_num_ses = int(row["num_ses"]) - num_removed
#     if new_num_ses < 0:
#         new_num_ses = 0

#     row["num_ses"] = new_num_ses
#     row["interval_days"] = format_interval_days(intervals_new)

#     print(f"\n  Updated num_ses: {row['num_ses']} (reduced by {num_removed})")
#     print(f"{'-'*80}")

#     return row


# def main():
#     # Load CSV
#     df = pd.read_csv(CSV_PATH)

#     # Process each patient row
#     updated_rows = []
#     for idx, row in df.iterrows():
#         updated_row = process_patient(row.copy())
#         updated_rows.append(updated_row)

#     # Create updated DataFrame and save
#     df_updated = pd.DataFrame(updated_rows)

#     print(f"Saving updated CSV to: {CSV_PATH}")
#     df_updated.to_csv(CSV_PATH, index=False)
#     print("\nDone.")


# if __name__ == "__main__":
#     main()

img = np.load('/l/users/alaa.mohamed/datasets/lumiere_proc/Patient-032_image.npy')
lbl = np.load('/l/users/alaa.mohamed/datasets/lumiere_proc/Patient-032_label.npy')

# Check shapes
print(f"Image: {img.shape}")  # e.g., (24, 240, 240, 155) = 4 modalities × 6 sessions
print(f"Label: {lbl.shape}")  # e.g., (6, 240, 240, 155) = 6 sessions

# Check value ranges
print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")  # Should be [0, 1]
print(f"Label values: {np.unique(lbl)}")  # Should be [0, 1, 3]

# Check for NaN/Inf
assert not np.isnan(img).any(), "NaN detected in images"
assert not np.isinf(img).any(), "Inf detected in images"

# ### Visualization

# # ------------------------------------------------------------------
# # Create figure: rows = sessions, columns = modalities (middle slice)
# # ------------------------------------------------------------------

# # Infer number of sessions (S) and modalities (C)
# S = lbl.shape[0]                 # number of sessions
# CS = img.shape[0]                # C * S
# assert CS % S == 0, "Image first dim is not divisible by number of sessions"
# C = CS // S                      # number of modalities (should be 4)

# print(f"Detected {S} sessions and {C} modalities.")

# # Reshape image to (S, C, H, W, D)
# H, W, D = img.shape[1:]
# img_reshaped = img.reshape(C, S, H, W, D)  # (C, S, H, W, D)
# img_reshaped = np.moveaxis(img_reshaped, 0, 1)  # -> (S, C, H, W, D)

# # Middle slice along the last axis (D)
# mid_slice = D // 2

# # Optional: names for modalities (if you know them; otherwise just use generic names)
# modality_names = [f"Modality {i+1}" for i in range(C)]

# # Create figure
# fig, axes = plt.subplots(
#     nrows=S,
#     ncols=C,
#     figsize=(3 * C, 3 * S),
#     squeeze=False
# )

# for s in range(S):
#     for c in range(C):
#         ax = axes[s, c]
#         slice2d = img_reshaped[s, c, :, :, mid_slice]

#         im = ax.imshow(slice2d, cmap='gray', vmin=0, vmax=1)
#         ax.axis('off')

#         # Titles for the first row only
#         if s == 0:
#             ax.set_title(modality_names[c], fontsize=10)

#         # Label rows with session index on the leftmost column
#         if c == 0:
#             ax.set_ylabel(f"Session {s+1}", fontsize=10)

# plt.tight_layout()

# # Save the figure
# out_path = "results/Patient-032_middle_slice_sessions_modalities.png"
# plt.savefig(out_path, dpi=300, bbox_inches='tight')
# plt.close(fig)

# print(f"Figure saved to: {out_path}")