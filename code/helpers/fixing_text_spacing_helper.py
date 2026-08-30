"""One-off helper: normalize whitespace and punctuation spacing in the corpus text."""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLEANED_CORPUS_CSV, DATA_INTERMEDIATE

# Load the file
file_path = CLEANED_CORPUS_CSV
df = pd.read_csv(file_path, encoding="utf-8-sig")


def clean_text(text):
    if pd.isna(text):
        return text

    # Remove newline characters within paragraphs
    text = re.sub(r"\n+", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s{2,}", " ", text)

    # Add space after punctuation if missing
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)

    # Optionally: separate concatenated words with a lowercase followed by uppercase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    return text.strip()


# Apply the cleaning function
df["Text"] = df["Text"].apply(clean_text)

# Optionally save the cleaned dataframe
df.to_csv(DATA_INTERMEDIATE / "corpus_cleaned_spacing.csv", index=False, encoding="utf-8-sig")
