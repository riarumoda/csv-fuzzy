import pandas as pd

def load_heart_data(csv_path):
    df = pd.read_csv(csv_path)
    # Selecting relevant columns
    df = df[['age', 'cp', 'trestbps', 'fbs', 'thalach', 'thal', 'target']]
    # Map thal: 3 → 0 (normal), 6 → 1 (fixed), 7 → 2 (reversible), if necessary
    df['thal'] = df['thal'].replace({3: 0, 6: 1, 7: 2})
    return df
