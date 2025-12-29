import pandas as pd

def main():
    df = pd.read_csv("LUMIERE-Demographics_Pathology.csv")
    df_patients = pd.read_csv("data/lumiere.csv")

    print("df.shape before filtering:", df.shape)

    # ---- Inspect MGMT qualitative ----
    print(df['MGMT qualitative'].value_counts(dropna=False))

    print("Missing MGMT qualitative:",
          df['MGMT qualitative'].isna().sum())

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
    
    df.to_csv("geno.csv")

if __name__ == "__main__":
    main()
