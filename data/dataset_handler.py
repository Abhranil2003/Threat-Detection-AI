
import os
import pandas as pd
import zipfile

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

def unzip_if_needed(zip_path: str, extract_to: str):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")

def process_cic_ids_2017():
    zip_path = os.path.join(RAW_DATA_DIR, "CIC-IDS-2017 Dataset.zip")
    extract_path = os.path.join(RAW_DATA_DIR, "CIC-IDS-2017")
    unzip_if_needed(zip_path, extract_path)

    processed_path = os.path.join(PROCESSED_DATA_DIR, "CIC-IDS-2017")
    os.makedirs(processed_path, exist_ok=True)

    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
                df.dropna(axis=1, how="all", inplace=True)
                cleaned_filename = os.path.splitext(file)[0] + "_cleaned.csv"
                df.to_csv(os.path.join(processed_path, cleaned_filename), index=False)
                print(f"Processed {file_path} -> {cleaned_filename}")

def process_nsl_kdd99():
    zip_path = os.path.join(RAW_DATA_DIR, "NSL-KDD99 Dataset.zip")
    extract_path = os.path.join(RAW_DATA_DIR, "NSL-KDD99")
    unzip_if_needed(zip_path, extract_path)

    processed_path = os.path.join(PROCESSED_DATA_DIR, "NSL-KDD99")
    os.makedirs(processed_path, exist_ok=True)

    col_names = [f"feature_{i}" for i in range(41)] + ["label", "difficulty"]

    for file in os.listdir(extract_path):
        if file.endswith(".txt") and ("KDDTrain" in file or "KDDTest" in file):
            file_path = os.path.join(extract_path, file)
            try:
                df = pd.read_csv(file_path, names=col_names)
                cleaned_filename = os.path.splitext(file)[0] + "_cleaned.csv"
                df.to_csv(os.path.join(processed_path, cleaned_filename), index=False)
                print(f"Processed {file_path} -> {cleaned_filename}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    process_cic_ids_2017()
    process_nsl_kdd99()
