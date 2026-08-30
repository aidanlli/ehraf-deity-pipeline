"""Scrape the paragraph text for every source permalink in the master list.

Step 04 of the scraping pipeline (~8 h). Fetches each row's eHRAF permalink
and appends "Raw Text" and "Text" columns. Progress is saved periodically so
the script can resume from an existing output file.
"""

import os
import sys
from pathlib import Path

import chardet
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONCATENATED_SOURCES_CSV, PARAGRAPH_TEXT_CSV

# File paths
input_path = CONCATENATED_SOURCES_CSV
output_path = PARAGRAPH_TEXT_CSV
save_interval = 1000  # Save every 1000 rows

# Load input CSV
df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str)

# Try loading existing output CSV to resume progress
if os.path.exists(output_path):
    print("Resuming from existing file...")
    df_existing = pd.read_csv(output_path)
    for col in ["Raw Text", "Text"]:
        if col in df_existing.columns:
            df[col] = df_existing[col]  # Preserve already scraped data
        else:
            df[col] = None  # Initialize missing columns
else:
    df["Raw Text"] = None
    df["Text"] = None

# User-Agent Header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

session = requests.Session()


# Function to scrape text
def scrape_text(url):
    try:
        with session.get(url, headers=headers, timeout=10) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # :one: First: Check for <span> with an id and store in Raw Text
            span_with_id = soup.find("span", id=True)
            raw_text = span_with_id.get_text(strip=True) if span_with_id else "No text found"

            # :two: Check for <span> with hraf-doc__sre--content class
            hraf_span = soup.find("span", class_="hraf-doc__sre--content")
            if hraf_span:
                return raw_text, hraf_span.get_text(strip=True)

            # :three: Check for <p> with specific class
            paragraph = soup.find("p", class_="mdc-typography--body2 undefined")
            if paragraph:
                return raw_text, paragraph.get_text(strip=True)

            # :four: If no <p>, check inside <span>
            span = soup.find("span", class_="mdc-typography--body2 linegroup")
            if span:
                return raw_text, span.get_text(strip=True)

            # :five: If no span, return default message
            return raw_text, "No text found"

    except requests.exceptions.HTTPError as e:
        if response.status_code == 451:
            return "Unavailable due to legal reasons", None
        return f"Error: {e}", None
    except requests.RequestException as e:
        return f"Error: {e}", None


# Scrape only missing rows
tqdm.pandas()
for i, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row["Text"]):  # Only scrape if Text column is empty
        raw_text, text = scrape_text(row["Permalink"])
        df.at[i, "Raw Text"] = raw_text
        df.at[i, "Text"] = text

    # Save progress every N rows
    if i % save_interval == 0:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

# Final save
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Updated CSV saved to {output_path}")

with open(output_path, "rb") as f:
    raw_data = f.read(10000)  # Read a portion of the file
    result = chardet.detect(raw_data)  # Detect encoding
    print(result)
