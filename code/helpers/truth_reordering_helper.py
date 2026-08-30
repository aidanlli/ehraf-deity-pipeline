"""One-off helper: reorder a rebuilt sample to match the truth table's row order."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_INTERMEDIATE, TRUTH_TABLE_CSV

# File paths
full_file_path = DATA_INTERMEDIATE / "250_sampled_rows_encoding.csv"
sampled_uuids_path = TRUTH_TABLE_CSV
output_file_path = DATA_INTERMEDIATE / "250_sampled_rows_reordered.csv"

# Load dataframes
df_full = pd.read_csv(full_file_path, encoding="utf-8-sig")
df_sampled = pd.read_csv(sampled_uuids_path, encoding="utf-8-sig")

# Ensure 'uuid' column exists
if "uuid" not in df_full.columns or "uuid" not in df_sampled.columns:
    raise ValueError("Missing 'uuid' column in one of the files.")

# Create a categorical type for 'uuid' in df_full to specify the order from df_sampled
df_full["uuid"] = pd.Categorical(df_full["uuid"], categories=df_sampled["uuid"], ordered=True)

# Sort df_full by the categorical order of 'uuid'
df_sorted = df_full.sort_values("uuid")

# Optionally, drop rows not in sampled UUIDs (if you want to keep only those UUIDs)
df_sorted = df_sorted.dropna(subset=["uuid"])

# Save the reordered dataframe
df_sorted.to_csv(output_file_path, index=False, encoding="utf-8-sig")

print(f"Sorted CSV saved to: {output_file_path}")
