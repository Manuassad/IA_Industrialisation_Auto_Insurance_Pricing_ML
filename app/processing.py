# Python file containing JSON payload processing functions for the auto pricing API.

# Import required libraries
import pandas as pd
import numpy as np

# Value used during training to group rare vehicle brands (fewer than 50 contracts)
RARE_BRAND_LABEL = 'Autre'

# Function to process a single JSON payload
def process_single_payload(payload: dict, known_brands: set[str] | None = None) -> pd.DataFrame:
    """
    Transforms a single payload dictionary (JSON) into an enriched DataFrame.

    Reproduces the feature engineering of the training script (training/step_final.py)
    for one contract at a time. The input dictionary is never modified.

    known_brands: brands seen by the fitted preprocessor. Any other brand is mapped to
    'Autre', exactly like the rare brands grouped during training.
    """
    df = pd.DataFrame([payload])

    # Nullable fields: JSON null (None) must become NaN, as in the training CSV.
    # Otherwise the imputers do not see a missing value: sex_conducteur2 = None would be
    # one-hot encoded as an unknown category instead of 'Inconnu'.
    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].astype(object).where(df['sex_conducteur2'].notna(), np.nan)
    if 'anciennete_vehicule' in df.columns:
        df['anciennete_vehicule'] = df['anciennete_vehicule'].astype(float)

    # Replace outliers / data entry errors (111 = unknown license seniority)
    for col in ['anciennete_permis1', 'anciennete_permis2']:
        if col in df.columns:
            df[col] = df[col].replace(111, np.nan)

    # Geographic feature extraction (handles Corsica: 2A / 2B)
    if 'code_postal' in df.columns:
        df['departement'] = df['code_postal'].astype(str).str.zfill(5).str[:2]

    # Rare brands were grouped into 'Autre' during training: apply the same rule
    if 'marque_vehicule' in df.columns:
        df['marque_vehicule'] = df['marque_vehicule'].astype(str).str.strip().str.upper()
        if known_brands is not None:
            df.loc[~df['marque_vehicule'].isin(known_brands), 'marque_vehicule'] = RARE_BRAND_LABEL

    # Interaction features (handling potential missing/null values)
    age1 = df.get('age_conducteur1', pd.Series([0]))
    permis1 = df.get('anciennete_permis1', pd.Series([0]))
    df['age_obtention_permis1'] = age1 - permis1
    df['is_jeune_permis'] = (permis1 < 3).astype(int)

    din = df.get('din_vehicule', pd.Series([0]))
    poids = df.get('poids_vehicule', pd.Series([0]))
    df['ratio_puissance_poids'] = din / (poids + 1)

    # A null vehicle age gives a null ratio, as in training (imputed by the median afterwards)
    prix = df.get('prix_vehicule', pd.Series([0]))
    anc_veh = df.get('anciennete_vehicule', pd.Series([np.nan]))
    df['ratio_valeur_anciennete'] = prix / (anc_veh + 1)

    age2 = df.get('age_conducteur2', pd.Series([0]))
    df['conducteur2_jeune'] = ((age2 > 0) & (age2 < 25)).astype(int)

    return df
