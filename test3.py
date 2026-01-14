import os
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("LUMIERE-Demographics_Pathology.csv")
    df_patients = pd.read_csv("data/lumiere.csv")

    print("df.shape before filtering:", df.shape)

    # ---- Inspect MGMT qualitative ----
    print(df['MGMT qualitative'].value_counts(dropna=False))

    print("Missing MGMT qualitative:",
          df['MGMT qualitative'].isna().sum())
    print(df[df['MGMT qualitative']=='na'])
    # ---- Keep only patients that exist in df_patients ----
    valid_patients = set(df_patients['patients'])

    df = df[df['Patient'].isin(valid_patients)].reset_index(drop=True)

    print("df.shape after filtering:", df.shape)

    # ---- MGMT qualitative → binary ----
    df['MGMT'] = df['MGMT qualitative'].map({
        'methylated': 1,
        'not methylated': 0
    })

    df = df[['Patient', 'MGMT qualitative']]


    print("df.shape after filtering:", df.shape)

    # ---- Inspect MGMT qualitative ----
    print(df['MGMT qualitative'].value_counts(dropna=False))

    print("Missing MGMT qualitative:",
          df['MGMT qualitative'].isna().sum())
    
    df.to_csv("geno2.csv")


def save_genomic_npy(
    csv_path: str,
    out_dir: str,
):
    # Load CSV
    df = pd.read_csv(csv_path)


    for _, row in df.iterrows():
        patient_id = row['Patient']
        print(patient_id)
        mgmt_value = row['MGMT']

        # Create 1D genomic vector
        geno_vec = np.array([mgmt_value], dtype=np.float32)

        # File name
        out_path = os.path.join(out_dir, f"{patient_id}_geno.npy")
        print(type(geno_vec), geno_vec.shape, geno_vec.dtype)
        np.save(out_path, geno_vec)

        print(f"Saved {out_path} with shape {geno_vec.shape}")

if __name__ == "__main__":
    main()
    # save_genomic_npy( csv_path="geno.csv", out_dir="/l/users/alaa.mohamed/datasets/lumiere_proc/" )
