"""
download_data.py
Downloads the Banknote Authentication Dataset from UCI ML Repository
and saves it to the data/ folder.
"""

import os
import urllib.request
import sys

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "banknote_authentication.csv")


def download_dataset():
    """Download the banknote authentication dataset from UCI."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print(f"✅ Dataset already exists at: {DATA_FILE}")
        return DATA_FILE

    print(f"⬇️  Downloading dataset from UCI ML Repository...")
    print(f"   URL: {DATA_URL}")

    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        
        # Add header row to the CSV (the original file has no headers)
        with open(DATA_FILE, "r") as f:
            content = f.read()
        
        header = "variance,skewness,kurtosis,entropy,class\n"
        with open(DATA_FILE, "w") as f:
            f.write(header + content)

        print(f"✅ Dataset downloaded successfully!")
        print(f"   Saved to: {DATA_FILE}")
        return DATA_FILE

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📋 Manual download instructions:")
        print(f"   1. Visit: {DATA_URL}")
        print(f"   2. Save the file as: {DATA_FILE}")
        print(f"   3. Add this header as the first line: variance,skewness,kurtosis,entropy,class")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
