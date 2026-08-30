"""One-off helper: rebuild the 250-row sample from the corrected-encoding corpus.

Used when the encoding of the sampled truth-table rows is wrong: pulls the
matching UUIDs back out of the final UTF-8 corpus.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_INTERMEDIATE, FINAL_CORPUS_CSV, TRUTH_TABLE_CSV

# File paths
processed_file = FINAL_CORPUS_CSV
sampled_file = TRUTH_TABLE_CSV
output_file = DATA_INTERMEDIATE / "250_sampled_rows_encoding.csv"

# Load the data
processed_df = pd.read_csv(processed_file, encoding="utf-8-sig")
sampled_df = pd.read_csv(sampled_file, encoding="utf-8-sig")

# Extract the uuids from sampled data
uuids_to_match = set(sampled_df["uuid"])

# Filter the processed data to match uuids
matched_rows = processed_df[processed_df["uuid"].isin(uuids_to_match)]

# Save the filtered rows to a new CSV
matched_rows.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved {len(matched_rows)} matched rows to {output_file}")
