from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from loader import load_data
from preprocess import ensure_datetime, extract_basic_features


def main():
    # Try to load data
    df = None
    try:
        df = load_data()
    except Exception as e:
        print(f"Load data failed: {e}", file=sys.stderr)
        sys.exit(1)
    # Basic preprocessing
    df = ensure_datetime(df)
    df = extract_basic_features(df)

    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    # Simple summary: show first few rows
    print(df.head())


if __name__ == "__main__":
    main()
