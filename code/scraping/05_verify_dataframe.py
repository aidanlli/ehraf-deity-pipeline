"""Verify the text-appended corpus against the concatenated source list.

Step 05 of the scraping pipeline. Confirms every paragraph UUID from step 02
survived step 04, and counts blank / "No text found" rows in the scraped
text columns.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONCATENATED_SOURCES_CSV, PARAGRAPH_TEXT_CSV

# File paths
input_path = CONCATENATED_SOURCES_CSV
output_path = PARAGRAPH_TEXT_CSV

# Load the data
concatenated_df = pd.read_csv(input_path)
updated_df = pd.read_csv(output_path)

# 1. Compare all uuid entries from concatenated and updated
concatenated_uuids = set(concatenated_df["uuid"])
updated_uuids = set(updated_df["uuid"])

# Check if all uuids in concatenated are in updated
missing_uuids = concatenated_uuids - updated_uuids
if not missing_uuids:
    print(
        "All uuids from paragraph source document are present in document with appended paragraphs."
    )
else:
    print(f"Missing uuids in updated output: {len(missing_uuids)}")
    print(missing_uuids)

# 2. Compare the number of rows in each dataframe
print(f"Number of rows in concatenated output: {len(concatenated_df)}")
print(f"Number of rows in updated output: {len(updated_df)}")

# 3. Remove rows in updated that don't have a matching uuid in concatenated
updated_df_filtered = updated_df[updated_df["uuid"].isin(concatenated_uuids)]
removed_rows = len(updated_df) - len(updated_df_filtered)
print(f"Number of rows removed from updated output: {removed_rows}")

# 4. Count rows with matching uuid that have blanks in 'Raw Text' and 'Text'
blanks_raw_text = updated_df_filtered["Raw Text"].isna().sum()
blanks_text = updated_df_filtered["Text"].isna().sum()
print(f"Rows with matching uuid having blanks in 'Raw Text': {blanks_raw_text}")
print(f"Rows with matching uuid having blanks in 'Text': {blanks_text}")

# 5. Count entries with "No text found" in 'Raw Text' and 'Text'
no_text_raw = (updated_df_filtered["Raw Text"] == "No text found").sum()
no_text_text = (updated_df_filtered["Text"] == "No text found").sum()
print(f"Entries with 'No text found' in 'Raw Text': {no_text_raw}")
print(f"Entries with 'No text found' in 'Text': {no_text_text}")
