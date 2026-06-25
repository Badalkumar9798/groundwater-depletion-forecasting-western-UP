"""
Step 1: Data Preparation
-------------------------
Source: CGWB (Central Ground Water Board) well-level water depth data, 1996-2017,
quarterly measurements (Jan/May/Aug/Nov), in metres below ground level (mbgl).
Higher value = water table is DEEPER = more depleted.

This script:
1. Loads the raw wide-format CSV (one row per well, one column per quarter)
2. Filters to Saharanpur + 5 neighbouring UP districts
3. Melts to long format and aggregates to a single district-level average per quarter
4. Interpolates small gaps so we get a continuous quarterly series
5. Saves a clean CSV used by the modeling step
"""

import pandas as pd
import numpy as np

RAW_PATH = "../data/CGWB_data_wide.csv"
OUT_PATH = "../output/district_quarterly_clean.csv"

DISTRICTS = ["Saharanpur", "Muzaffarnagar", "Meerut", "Bijnor", "Bareilly", "Moradabad"]

def main():
    df = pd.read_csv(RAW_PATH)
    date_cols = df.columns[7:]  # all the quarter columns, e.g. "May 1996"

    sub = df[df["DISTRICT"].isin(DISTRICTS)].copy()
    print(f"Wells found: {sub.groupby('DISTRICT').size().to_dict()}")

    # Melt wide -> long
    long_df = sub.melt(
        id_vars=["DISTRICT", "LAT", "LON", "WLCODE"],
        value_vars=date_cols,
        var_name="quarter_label",
        value_name="depth_m"
    )
    long_df = long_df.dropna(subset=["depth_m"])

    # Parse "May 1996" style labels into actual dates (use mid-quarter month)
    long_df["date"] = pd.to_datetime(long_df["quarter_label"], format="%b %Y")

    # Aggregate: mean depth across all wells in a district for that quarter
    agg = (
        long_df.groupby(["DISTRICT", "date"])["depth_m"]
        .agg(mean_depth="mean", well_count="count")
        .reset_index()
        .sort_values(["DISTRICT", "date"])
    )

    # Build a full quarterly date range per district and interpolate small gaps
    # (CGWB nominal quarters: Jan, May, Aug, Nov)
    full_rows = []
    for district, grp in agg.groupby("DISTRICT"):
        grp = grp.set_index("date").sort_index()
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="QS")
        # snap to nearest available quarter rather than forcing exact Jan/May/Aug/Nov
        grp = grp.reindex(grp.index.union(full_idx)).sort_index()
        grp["mean_depth"] = grp["mean_depth"].interpolate(method="linear", limit=2)
        grp["DISTRICT"] = district
        grp = grp.dropna(subset=["mean_depth"])
        grp = grp.reset_index().rename(columns={"index": "date"})
        full_rows.append(grp[["DISTRICT", "date", "mean_depth", "well_count"]])

    clean = pd.concat(full_rows, ignore_index=True)
    clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved clean dataset: {OUT_PATH}")
    print(f"Rows: {len(clean)}, Districts: {clean['DISTRICT'].nunique()}")
    print(clean.groupby("DISTRICT")["date"].agg(["min", "max", "count"]))

if __name__ == "__main__":
    main()
