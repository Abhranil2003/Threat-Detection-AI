import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] Loaded data shape: {data.shape}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def clean_data(df):
    """Handle missing values and drop irrelevant columns."""
    df = df.copy()
    df.dropna(inplace=True)
    
    # Optional: drop columns like 'Flow ID', 'Timestamp', if they exist
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    print(f"[INFO] Data shape after cleaning: {df.shape}")
    return df

def encode_features(df):
    """Encode categorical labels using LabelEncoder."""
    df = df.copy()
    encoders = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"[INFO] Encoded {col}")

    return df, encoders

def scale_features(X):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[INFO] Feature scaling complete.")
    return X_scaled, scaler

def preprocess_pipeline(input_path, output_path, label_column="Label"):
    """Full preprocessing pipeline."""
    df = load_data(input_path)
    if df is None:
        return

    df = clean_data(df)
    df, encoders = encode_features(df)

    if label_column not in df.columns:
        raise ValueError(f"[ERROR] Label column '{label_column}' not found.")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save processed datasets
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "X_train.npy"), X_train)
    np.save(os.path.join(output_path, "X_test.npy"), X_test)
    np.save(os.path.join(output_path, "y_train.npy"), y_train)
    np.save(os.path.join(output_path, "y_test.npy"), y_test)

    # Save encoders and scaler
    joblib.dump(scaler, os.path.join(output_path, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(output_path, "encoders.pkl"))

    print(f"[INFO] Preprocessing complete. Files saved in '{output_path}'.")

# Example usage
if __name__ == "__main__":
    preprocess_pipeline(
        input_path="data/raw/your_dataset.csv",
        output_path="data/processed",
        label_column="Label"
    )
