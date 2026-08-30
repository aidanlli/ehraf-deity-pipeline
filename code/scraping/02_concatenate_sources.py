"""Concatenate all per-culture eHRAF export CSVs into one master dataframe.

Step 02 of the scraping pipeline. Reads every CSV in
``data/intermediate/ehraf_exports/`` (the step 01 downloads plus the manual
exports from ``data/raw/manual_culture_exports/``), keeps the expected
columns, drops duplicate paragraph UUIDs, and writes one combined file.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONCATENATED_SOURCES_CSV, SCRAPED_EXPORTS_DIR


def count_and_concatenate(folder_path):
    # Define the expected columns
    expected_columns = [
        "uuid",
        "Primary_Author",
        "Title",
        "Published",
        "Page",
        "DocType",
        "Culture",
        "Region",
        "Subregion",
        "Subsistence",
        "OCM",
        "IDs",
        "Permalink",
    ]

    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # Count files
    file_count = len(files)
    print(f"Total files found: {file_count}")

    # Read and concatenate all files
    df_list = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Ensure the file contains the expected columns
        if set(expected_columns).issubset(df.columns):
            df_list.append(df[expected_columns])
        else:
            print(f"Warning: {file} does not have the expected columns.")

    # Concatenate all DataFrames
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        print("Concatenation successful.")
        initial_count = len(final_df)
        final_df.drop_duplicates(subset="uuid", keep="first", inplace=True)
        duplicates_removed = initial_count - len(final_df)
        print(f"Duplicates removed: {duplicates_removed} (based on 'uuid' column).")
        return final_df
    else:
        print("No valid files to concatenate.")
        return None


concatenated_df = count_and_concatenate(SCRAPED_EXPORTS_DIR)
if concatenated_df is not None:
    concatenated_df.to_csv(CONCATENATED_SOURCES_CSV, index=False)
    print(f"Concatenated file saved to {CONCATENATED_SOURCES_CSV}")
