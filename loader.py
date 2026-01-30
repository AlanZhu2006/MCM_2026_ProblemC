import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def load_data(path: str | None = None) -> pd.DataFrame:
    # Try a few common default locations
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        os.path.join(os.getcwd(), "2026_MCM_Problem_C_Data.csv"),
        os.path.join(
            os.getcwd(), "2026_MCM-ICM_Problems", "2026_MCM_Problem_C_Data.csv"
        ),
        os.path.join(
            os.getcwd(), "2026_MCM-ICM_Problems", "data", "2026_MCM_Problem_C_Data.csv"
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        "Could not locate 2026_MCM_Problem_C_Data.csv in known locations: "
        + ", ".join(candidates)
    )
