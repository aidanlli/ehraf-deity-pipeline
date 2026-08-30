"""Re-encode the cleaned corpus as UTF-8, repairing latin-1 mojibake.

Step 08 of the scraping pipeline. Fixes double-encoded characters in the
text columns and writes the final UTF-8 corpus, then spot-checks one known
row (the phrase "kikinè" should render correctly).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLEANED_CORPUS_CSV, FINAL_CORPUS_CSV

# File paths
input_path = CLEANED_CORPUS_CSV
output_path = FINAL_CORPUS_CSV

# Load the CSV file into a DataFrame
df = pd.read_csv(input_path)


def fix_encoding(text):
    if isinstance(text, str):  # Ensure it's a string before processing
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeEncodeError:
            return text  # If it fails, return original text
    return text


# Apply to both "Raw Text" and "Text" columns
df["Raw Text"] = df["Raw Text"].apply(fix_encoding)
df["Text"] = df["Text"].apply(fix_encoding)

# Save the fixed CSV
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Encoding fixed and saved to {output_path}")


df2 = pd.read_csv(output_path)

# Define the UUID to search for
uuid_to_find = "fa5057ed-1b89-4f3c-8156-0a2eb363e6be"

# Find the row where the "uuid" column matches the given value
matching_row = df2[df2["uuid"] == uuid_to_find]

# Extract the value from the "Text" column
if not matching_row.empty:
    text_value = matching_row["Text"].values[0]
    print("Text value:", text_value)
else:
    print("UUID not found in the file.")
