"""Row-level TP/FP/FN and text-length metrics for pipeline result CSVs.

Adds per-row columns (match counts, word/character rates, TP/FP/FN
breakdowns) to a results file for bulk analysis. Formerly ``elongation.py``.
"""

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_INTERMEDIATE


def normalize_name(name: str, keep_missing: bool = True) -> str:
    """Normalize deity names for comparison."""
    if pd.isna(name):
        return "missing" if keep_missing else ""
    name = str(name).strip().strip('"').lower()
    return name if name else ("missing" if keep_missing else "")


def is_missing_or_blank(value) -> bool:
    """Check if a value is missing, blank, or the string 'missing'."""
    if pd.isna(value):
        return True
    if isinstance(value, str):
        cleaned = value.strip().strip('"').lower()
        return cleaned == "missing" or cleaned == ""
    return False


def parse_comma_column(column):
    """Parse comma-separated column values."""
    if pd.isna(column) or column == "missing":
        return []

    if isinstance(column, str):
        stripped = column.strip()

        # Handle list format like [1, 0] or ['item1', 'item2']
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                evaluated = ast.literal_eval(stripped)
                if isinstance(evaluated, list):
                    return [
                        normalize_name(str(item), keep_missing=True)
                        for item in evaluated
                        if item not in [0, "0"]
                    ]
            except (ValueError, SyntaxError):
                pass

        # Handle quoted comma-separated strings like '"deity1", "deity2"'
        # Split by comma and clean up quotes
        parts = []
        for part in stripped.split(","):
            cleaned = part.strip().strip('"').strip("'")
            if cleaned and cleaned.lower() != "missing":
                parts.append(normalize_name(cleaned, keep_missing=True))
        return parts

    return []


def safe_eval(value: str) -> Any:
    """Safely evaluate string representations of Python literals."""
    if pd.isna(value) or value == "missing":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def count_deities_from_string(deity_string: str) -> int:
    """Count deities from a string representation."""
    if pd.isna(deity_string) or deity_string == "missing":
        return 0

    # Handle quoted deity names separated by commas
    # Remove outer quotes and split by comma, then count non-empty items
    cleaned = deity_string.strip('"')
    if cleaned == "missing":
        return 0

    deities = [d.strip().strip('"') for d in cleaned.split(",")]
    return len([d for d in deities if d and d != "missing"])


def count_deities_from_dict_or_list(data: dict | list | str) -> int:
    """Count deities from dictionary or list representation."""
    if data is None or data == "missing":
        return 0

    if isinstance(data, str):
        return count_deities_from_string(data)
    elif isinstance(data, dict):
        return len(data)
    elif isinstance(data, list):
        return len([item for item in data if item not in [0, "0"]])
    else:
        return 0


def calculate_match_percentage(
    matched_deities: dict | str, truth_count: int, final_count: int, both_missing: bool = False
) -> float:
    """Calculate percentage of deities that were successfully matched."""
    if both_missing:
        return np.nan

    if truth_count == 0:
        return 0.0

    if isinstance(matched_deities, str):
        matched_data = safe_eval(matched_deities)
    else:
        matched_data = matched_deities

    if not isinstance(matched_data, dict):
        return 0.0

    # Count successful matches (score > 0)
    successful_matches = sum(
        1
        for match_info in matched_data.values()
        if isinstance(match_info, dict) and match_info.get("score", 0) > 0
    )

    return (successful_matches / truth_count) * 100 if truth_count > 0 else 0.0


def calculate_tp_fp_fn(row) -> tuple:
    """Calculate True Positives, False Positives, and False Negatives for a single row."""
    # Check if both truth_deities and final_deities are missing
    if is_missing_or_blank(row["truth_deities"]) and is_missing_or_blank(row["final_deities"]):
        return np.nan, np.nan, np.nan

    TP, FP, FN = 0, 0, 0

    # Safely parse matched_deities
    if isinstance(row["matched_deities"], str):
        try:
            matched_deities = ast.literal_eval(row["matched_deities"])
        except (ValueError, SyntaxError):
            matched_deities = {}
    else:
        matched_deities = row["matched_deities"]

    if not isinstance(matched_deities, dict):
        return 0, 0, 0

    used_final_deities = set()

    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info.get("match", "missing")
            score = match_info.get("score", 0)
        else:
            match, score = match_info if isinstance(match_info, tuple) else ("missing", 0)

        if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
            # This is a correct identification of missing
            continue
        elif score >= 50:
            TP += 1
            # Add the normalized matched deity name to used set
            used_final_deities.add(normalize_name(match, keep_missing=True))
        else:
            if str(row["truth_deities"]).strip().lower() != "missing":
                FN += 1

    # Count false positives (final deities that weren't successfully matched)
    final_deities = parse_comma_column(row["final_deities"])

    unmatched_predictions = [
        d for d in final_deities if d not in used_final_deities and d != "missing"
    ]
    FP = len(unmatched_predictions)

    return TP, FP, FN


def count_words(text: str) -> int:
    """Count words in text."""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text).split())


def count_characters(text: str) -> int:
    """Count characters in text."""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text))


def calculate_gender_match_percentage(
    truth_genders: str, final_genders: str, both_missing: bool = False
) -> float:
    """Calculate percentage of gender matches."""
    if both_missing:
        return np.nan

    if pd.isna(truth_genders) or pd.isna(final_genders) or truth_genders == "missing":
        return 0.0

    # Parse gender strings - they appear to be comma-separated
    truth_list = [g.strip() for g in truth_genders.split(",")]
    final_list = [g.strip() for g in final_genders.split(",")]

    # Remove 'missing' entries
    truth_list = [g for g in truth_list if g != "missing"]
    final_list = [g for g in final_list if g != "missing"]

    if len(truth_list) == 0:
        return 0.0

    # Count matches
    matches = 0
    for i, truth_gender in enumerate(truth_list):
        if i < len(final_list) and truth_gender == final_list[i]:
            matches += 1

    return (matches / len(truth_list)) * 100


def process_csv(input_file: str, output_file: str = None) -> pd.DataFrame:
    """Process the CSV file and add the new columns."""

    # Try different encodings to read the CSV
    encodings = ["utf-8-sig"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully read CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("Could not read CSV file with any of the attempted encodings")

    print(f"Processing {len(df)} rows...")

    # Check if both truth_deities and final_deities are missing for each row
    df["both_missing"] = df.apply(
        lambda row: (
            is_missing_or_blank(row["truth_deities"]) and is_missing_or_blank(row["final_deities"])
        ),
        axis=1,
    )

    # Add truth_deities_count column
    df["truth_deities_count"] = df.apply(
        lambda row: (
            np.nan
            if row["both_missing"]
            else count_deities_from_string(row["truth_deities"])
            if isinstance(row["truth_deities"], str)
            else 0
        ),
        axis=1,
    )

    # Add final_deities_count column
    df["final_deities_count"] = df.apply(
        lambda row: (
            np.nan
            if row["both_missing"]
            else count_deities_from_string(row["final_deities"])
            if isinstance(row["final_deities"], str)
            else 0
        ),
        axis=1,
    )

    # Add %matched column
    df["percent_matched"] = df.apply(
        lambda row: calculate_match_percentage(
            row["matched_deities"],
            row["truth_deities_count"] if not pd.isna(row["truth_deities_count"]) else 0,
            row["final_deities_count"] if not pd.isna(row["final_deities_count"]) else 0,
            row["both_missing"],
        ),
        axis=1,
    )

    # Add word count and character count for Text column
    if "Text" in df.columns:
        df["word_count"] = df["Text"].apply(count_words)
        df["character_count"] = df["Text"].apply(count_characters)
        print("Added word_count and character_count columns for Text column")

    # Calculate TP, FP, FN for each row
    tp_fp_fn_results = df.apply(calculate_tp_fp_fn, axis=1)
    df["true_positives"] = [result[0] for result in tp_fp_fn_results]
    df["false_positives"] = [result[1] for result in tp_fp_fn_results]
    df["false_negatives"] = [result[2] for result in tp_fp_fn_results]

    # Calculate per-character and per-word rates
    if "Text" in df.columns:
        # Per character rates (avoid division by zero and handle NaN)
        df["tp_per_character"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["true_positives"])
                else (
                    row["true_positives"] / row["character_count"]
                    if row["character_count"] > 0
                    else 0
                )
            ),
            axis=1,
        )
        df["fp_per_character"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["false_positives"])
                else (
                    row["false_positives"] / row["character_count"]
                    if row["character_count"] > 0
                    else 0
                )
            ),
            axis=1,
        )
        df["fn_per_character"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["false_negatives"])
                else (
                    row["false_negatives"] / row["character_count"]
                    if row["character_count"] > 0
                    else 0
                )
            ),
            axis=1,
        )

        # Per word rates (avoid division by zero and handle NaN)
        df["tp_per_word"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["true_positives"])
                else (row["true_positives"] / row["word_count"] if row["word_count"] > 0 else 0)
            ),
            axis=1,
        )
        df["fp_per_word"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["false_positives"])
                else (row["false_positives"] / row["word_count"] if row["word_count"] > 0 else 0)
            ),
            axis=1,
        )
        df["fn_per_word"] = df.apply(
            lambda row: (
                np.nan
                if pd.isna(row["false_negatives"])
                else (row["false_negatives"] / row["word_count"] if row["word_count"] > 0 else 0)
            ),
            axis=1,
        )

    # Calculate correct rate: TP / (TP + FP + FN)
    df["correct_rate"] = df.apply(
        lambda row: (
            np.nan
            if pd.isna(row["true_positives"])
            else (
                row["true_positives"]
                / (row["true_positives"] + row["false_positives"] + row["false_negatives"])
                if (row["true_positives"] + row["false_positives"] + row["false_negatives"]) > 0
                else 0
            )
        ),
        axis=1,
    )

    # Add %gender_matched column
    df["percent_gender_matched"] = df.apply(
        lambda row: calculate_gender_match_percentage(
            row["truth_genders"], row["final_genders"], row["both_missing"]
        ),
        axis=1,
    )

    # Round rates to 4 decimal places for better readability (only for non-NaN values)
    if "Text" in df.columns:
        df["tp_per_character"] = df["tp_per_character"].round(4)
        df["fp_per_character"] = df["fp_per_character"].round(4)
        df["fn_per_character"] = df["fn_per_character"].round(4)
        df["tp_per_word"] = df["tp_per_word"].round(4)
        df["fp_per_word"] = df["fp_per_word"].round(4)
        df["fn_per_word"] = df["fn_per_word"].round(4)

    df["correct_rate"] = df["correct_rate"].round(4)
    df["percent_matched"] = df["percent_matched"].round(2)
    df["percent_gender_matched"] = df["percent_gender_matched"].round(2)

    # Drop the temporary 'both_missing' column
    df = df.drop("both_missing", axis=1)

    # Display summary statistics (excluding NaN values)
    print("\nSummary Statistics:")
    print(
        f"Rows with both truth_deities and final_deities missing: {df['truth_deities_count'].isna().sum()}"
    )
    print(f"Average truth deities count: {df['truth_deities_count'].mean():.2f}")
    print(f"Average final deities count: {df['final_deities_count'].mean():.2f}")
    print(f"Average match percentage: {df['percent_matched'].mean():.2f}%")
    print(f"Average gender match percentage: {df['percent_gender_matched'].mean():.2f}%")
    print(f"Average correct rate: {df['correct_rate'].mean():.4f}")

    # Text statistics if available
    if "Text" in df.columns:
        print(f"Average word count: {df['word_count'].mean():.2f}")
        print(f"Average character count: {df['character_count'].mean():.2f}")
        print(f"Average TP per word: {df['tp_per_word'].mean():.4f}")
        print(f"Average FP per word: {df['fp_per_word'].mean():.4f}")
        print(f"Average FN per word: {df['fn_per_word'].mean():.4f}")
        print(f"Average TP per character: {df['tp_per_character'].mean():.4f}")
        print(f"Average FP per character: {df['fp_per_character'].mean():.4f}")
        print(f"Average FN per character: {df['fn_per_character'].mean():.4f}")

    # TP/FP/FN statistics (excluding NaN values)
    total_tp = df["true_positives"].sum()
    total_fp = df["false_positives"].sum()
    total_fn = df["false_negatives"].sum()

    print("\nDeity Detection Performance:")
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")

    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"Precision: {precision:.3f}")

    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Recall: {recall:.3f}")

    if total_tp + total_fp + total_fn > 0:
        overall_correct_rate = total_tp / (total_tp + total_fp + total_fn)
        print(f"Overall Correct Rate: {overall_correct_rate:.4f}")

    # Show sample of new columns
    print("\nSample of new columns:")
    new_columns = [
        "truth_deities_count",
        "final_deities_count",
        "percent_matched",
        "percent_gender_matched",
        "true_positives",
        "false_positives",
        "false_negatives",
        "correct_rate",
    ]

    # Add text-based columns if they exist
    if "Text" in df.columns:
        new_columns.extend(
            [
                "word_count",
                "character_count",
                "tp_per_word",
                "fp_per_word",
                "fn_per_word",
                "tp_per_character",
                "fp_per_character",
                "fn_per_character",
            ]
        )

    available_columns = [col for col in new_columns if col in df.columns]
    print(df[available_columns].head(10))

    # Save to output file if specified
    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nProcessed data saved to: {output_file}")

    return df


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path - use one of these methods:

    input_filename = DATA_INTERMEDIATE / "deity_matching_results.csv"
    output_filename = DATA_INTERMEDIATE / "deity_matching_results_processed.csv"

    try:
        processed_df = process_csv(input_filename, output_filename)
        print("\nProcessing completed successfully!")

        # Optionally, display the first few rows with all columns
        print("\nFirst few rows with new columns:")
        print(processed_df.head())

    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        print("Please update the input_filename variable with your actual CSV file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
