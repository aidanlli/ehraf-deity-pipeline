"""Check that every eHRAF culture is present in the concatenated source list.

Step 03 of the scraping pipeline. Reports (and saves) any cultures from the
master eHRAF culture list that are missing from the step 02 output so they
can be re-scraped.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    CONCATENATED_SOURCES_CSV,
    CULTURE_SUMMARY_CSV,
    MISSING_CULTURES_CSV,
)

# File paths
csv_path = CULTURE_SUMMARY_CSV
concatenated_csv_path = CONCATENATED_SOURCES_CSV

# Load the data
df_summary = pd.read_csv(csv_path)
df_concatenated = pd.read_csv(concatenated_csv_path)

# Extract unique culture names (strip spaces for consistency)
summary_cultures = set(df_summary["EHRAF WORLD CULTURES NAME"].dropna().str.strip())
concatenated_cultures = set(df_concatenated["Culture"].dropna().str.strip())

# Find cultures in qrySummary but NOT in concatenated_output
missing_in_concatenated = summary_cultures - concatenated_cultures

# Save the missing cultures to a CSV file
missing_cultures_path = MISSING_CULTURES_CSV
pd.DataFrame({"Missing Cultures": list(missing_in_concatenated)}).to_csv(
    missing_cultures_path, index=False
)

# Print the result - we should see the following: ['Hazara', 'Pamir Peoples', 'Turkmens', 'Dominicans', 'Eastern Apache']
print(f"Total missing cultures: {len(missing_in_concatenated)}")
print(f"Missing cultures saved to: {missing_cultures_path}")
for culture in missing_in_concatenated:
    print(f"Missing culture: {culture}")
print(list(missing_in_concatenated))
