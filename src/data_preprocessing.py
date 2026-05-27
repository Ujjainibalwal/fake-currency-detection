"""
data_preprocessing.py
Handles loading, exploring, and preprocessing the banknote dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Path to the dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "banknote_authentication.csv")


def load_data(filepath=None):
    """
    Load the banknote authentication dataset.
    
    Parameters:
        filepath (str): Path to the CSV file. Defaults to data/banknote_authentication.csv
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    if filepath is None:
        filepath = DATA_FILE

    if not os.path.exists(filepath):
        print(f"❌ Dataset not found at: {filepath}")
        print("   Run 'python src/download_data.py' first to download the dataset.")
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Parameters:
        df (pd.DataFrame): The dataset
    """
    print("=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Basic info
    print("\n📋 Dataset Info:")
    print(f"   Total samples    : {df.shape[0]}")
    print(f"   Total features   : {df.shape[1] - 1}")
    print(f"   Feature names    : {list(df.columns[:-1])}")
    print(f"   Target column    : '{df.columns[-1]}'")

    # Class distribution
    print("\n📈 Class Distribution:")
    class_counts = df["class"].value_counts()
    print(f"   Genuine (0) : {class_counts.get(0, 0)} samples ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Forged  (1) : {class_counts.get(1, 0)} samples ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    # Missing values
    print(f"\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found ✅")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count} missing")

    # Statistical summary
    print(f"\n📉 Statistical Summary:")
    print(df.describe().round(4).to_string())
    print()

    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset: split into train/test and scale features.
    
    Parameters:
        df (pd.DataFrame): The dataset
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("=" * 60)
    print("⚙️  DATA PREPROCESSING")
    print("=" * 60)

    # Separate features and target
    X = df.drop("class", axis=1)
    y = df["class"]

    # Train-test split (stratified to maintain class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n✂️  Train-Test Split ({int((1-test_size)*100)}/{int(test_size*100)}):")
    print(f"   Training set : {X_train.shape[0]} samples")
    print(f"   Testing set  : {X_test.shape[0]} samples")

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames for convenience
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    print(f"\n📏 Feature Scaling (StandardScaler):")
    print(f"   Mean (after scaling) : {X_train_scaled.mean().values.round(6)}")
    print(f"   Std  (after scaling) : {X_train_scaled.std().values.round(4)}")
    print(f"   ✅ Features standardized (mean≈0, std≈1)\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Run as standalone to test preprocessing
    df = load_data()
    explore_data(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("✅ Preprocessing complete!")
