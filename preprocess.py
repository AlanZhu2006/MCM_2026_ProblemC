import pandas as pd


def ensure_datetime(df: pd.DataFrame, date_col_candidates=None) -> pd.DataFrame:
    if date_col_candidates is None:
        date_col_candidates = ["Date", "date", "date_posted", "date_time"]
    for c in date_col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.rename(columns={c: "date"})
            break
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(pd.Series([pd.Timestamp("now")] * len(df)))
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    return df


def extract_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Simple per-row feature: text length etc.
    df = df.copy()
    if "Title/Content" in df.columns:
        df["content_len"] = df["Title/Content"].astype(str).str.len()
    elif "Content" in df.columns:
        df["content_len"] = df["Content"].astype(str).str.len()
    else:
        df["content_len"] = 0
    return df
