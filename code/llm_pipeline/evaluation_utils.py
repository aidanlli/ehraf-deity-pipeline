"""
Analysis Utilities for Deity Matching and Evaluation

This module contains utility functions for analyzing deity identification results,
including deity matching, evaluation metrics, gender analysis, and visualization.

Functions extracted from:
- chatgpt_truth_comparison.py
- scoring_gpt_v2.py

Functions:
    Data Processing:
    - normalize_name: Normalize deity names for comparison
    - parse_comma_column: Parse comma-separated columns
    - parse_linked_column: Parse semicolon-separated columns
    - extract_matched: Extract matched deities from dictionary format

    Deity Matching:
    - match_deities_with_aliases: Match deities using fuzzy matching with aliases
    - evaluate_row: Evaluate a single row for deity matching accuracy

    Metrics & Evaluation:
    - evaluate_subset: Calculate TP, FP, FN, precision, recall, F1 for a subset
    - evaluate_gender_matches_wide: Evaluate gender matching accuracy
    - evaluate_gender_accuracy_wide: Evaluate gender prediction accuracy for matched deities
    - calculate_fp_fn_tp: Calculate false positives, false negatives, and true positives per row
    - count_cat_types: Count individual vs multiple deity types

    Visualization:
    - save_df_as_png: Save dataframe as PNG image
    - plot_category_comparison: Plot category comparisons with error bars
    - plot_regional_metrics: Plot FP/FN metrics per region/subregion

Author: Aidan Li
Date: January 2026
"""

import ast
from collections import Counter, defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fuzzywuzzy import fuzz, process
from scipy.stats import norm

# ==============================================================================
# DATA NORMALIZATION AND PARSING FUNCTIONS
# ==============================================================================


def normalize_name(name: Any, keep_missing: bool = False) -> str | None:
    """
    Normalize deity names for comparison.

    Args:
        name: Input name (can be string, number, or None)
        keep_missing (bool): If True, preserve "missing" as a value

    Returns:
        Optional[str]: Normalized name in lowercase, or None
    """
    if pd.isna(name):
        return None
    s = str(name).strip()
    s_lower = s.lower()

    if not s_lower:
        return None
    if s_lower in {"missing", "na"}:
        return s_lower if keep_missing else None

    return s_lower


def parse_comma_column(column: Any) -> list[str | None]:
    """
    Parse comma-separated column values.

    Handles both list-formatted strings (e.g., "['a', 'b']") and
    plain comma-separated strings (e.g., "a, b").

    Args:
        column: Column value to parse

    Returns:
        List[Optional[str]]: List of normalized values
    """
    if isinstance(column, str):
        stripped = column.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                evaluated = ast.literal_eval(stripped)
                if isinstance(evaluated, list):
                    return [normalize_name(item, keep_missing=True) for item in evaluated]
            except (ValueError, SyntaxError):
                pass
        return [normalize_name(part, keep_missing=True) for part in column.split(",")]
    return []


def parse_linked_column(column: Any) -> list[str | None]:
    """
    Parse semicolon-separated column values.

    Args:
        column: Column value to parse

    Returns:
        List[Optional[str]]: List of normalized values
    """
    return [normalize_name(part) for part in column.split(";")] if isinstance(column, str) else []


def extract_matched(d: Any) -> list[str]:
    """
    Extract matched deities from dictionary format.

    Args:
        d: Dictionary string or object containing matched deity information

    Returns:
        List[str]: List of matched deity names with score > 0
    """
    try:
        parsed = ast.literal_eval(d)
        if isinstance(parsed, dict):
            return [
                k.strip('"').strip("'")
                for k, v in parsed.items()
                if isinstance(v, dict) and v.get("score", 0) > 0
            ]
        return []
    except Exception:
        return []


# ==============================================================================
# DEITY MATCHING FUNCTIONS
# ==============================================================================


def match_deities_with_aliases(
    deity_aliases: dict[str, list[str]], final_deities: list[str]
) -> dict[str, dict[str, Any]]:
    """
    Match deities using fuzzy matching with aliases.

    Implements a greedy one-to-one matching algorithm:
    1. Collect all possible matches with scores
    2. Sort by score (descending)
    3. Assign matches ensuring each deity is matched only once

    Args:
        deity_aliases: Dictionary mapping base deity names to lists of aliases
        final_deities: List of deities to match against

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of base deities to match information
            {base_deity: {"match": matched_name, "score": score, "matched_via": alias}}
    """
    candidate_matches = []

    # Step 1: Collect all possible matches with scores
    for base_deity, aliases in deity_aliases.items():
        for alias in aliases:
            match, score = process.extractOne(
                alias, final_deities, scorer=fuzz.token_sort_ratio
            ) or (None, 0)
            if match:
                candidate_matches.append(
                    {"base_deity": base_deity, "alias": alias, "match": match, "score": score}
                )

    # Step 2: Sort matches by score descending
    candidate_matches.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Assign matches greedily ensuring one-to-one matching
    used_final_deities = set()
    matched_final_set = set()
    matches = {}

    for candidate in candidate_matches:
        base = candidate["base_deity"]
        match = candidate["match"]
        if base in matches:
            continue  # already matched
        if match in used_final_deities:
            continue  # already used by another base_deity
        matches[base] = {
            "match": match,
            "score": candidate["score"],
            "matched_via": candidate["alias"],
        }
        used_final_deities.add(match)
        matched_final_set.add(match)

    # Step 4: Fill in unmatched deities as "missing"
    for base in deity_aliases:
        if base not in matches:
            matches[base] = {"match": "missing", "score": 0, "matched_via": None}

    return matches


def evaluate_row(uuid: str, truth_row: pd.Series, final_row: pd.Series) -> dict[str, Any]:
    """
    Evaluate deity matching for a single row.

    Compares ground truth deities against predicted deities, calculating
    match scores and gender accuracy.

    Args:
        uuid (str): Unique identifier for the row
        truth_row (pd.Series): Ground truth data row
        final_row (pd.Series): Predicted data row

    Returns:
        Dict[str, Any]: Evaluation results including matched deities, scores, and gender scores
    """
    truth_deities = parse_comma_column(truth_row.get("Deities", ""))
    truth_genders = parse_comma_column(truth_row.get("Gender", ""))
    other_names = parse_linked_column(truth_row.get("Other_names", ""))

    final_deities = parse_comma_column(final_row.get("deities", ""))
    final_genders = parse_comma_column(final_row.get("gender", ""))
    other_names_final = parse_linked_column(final_row.get("other names", ""))

    deity_aliases = {}
    for i, deity in enumerate(truth_deities):
        if deity is None:
            continue
        alias = other_names[i] if i < len(other_names) else None
        deity_aliases[deity] = [deity] + ([alias] if alias else [])

    # Handle case where both have no deities
    if all(d is None for d in truth_deities) and all(d is None for d in final_deities):
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get("Deities", ""),
            "final_deities": final_row.get("deities", ""),
            "matched_deities": str({"missing": ("missing", 100)}),
            "score": [1],
            "gender_score": [1],
            "truth_genders": truth_row.get("Gender", ""),
            "final_genders": final_row.get("gender", ""),
        }

    # Handle case where truth has no deities but final does (false positives)
    elif all(d is None for d in truth_deities):
        unmatched = [d for d in final_deities if d]
        score = [0 for _ in unmatched]
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get("Deities", ""),
            "final_deities": final_row.get("deities", ""),
            "matched_deities": str({"missing": (", ".join(unmatched), 0)}),
            "score": score,
            "gender_score": [0] * len(score),
            "truth_genders": truth_row.get("Gender", ""),
            "final_genders": final_row.get("gender", ""),
        }

    # Handle case where truth has deities but final doesn't (false negatives)
    elif any(d is not None for d in truth_deities) and all(d is None for d in final_deities):
        score = [0 for d in truth_deities if d is not None]
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get("Deities", ""),
            "final_deities": final_row.get("deities", ""),
            "matched_deities": str({d: ("missing", 0) for d in truth_deities if d}),
            "score": score,
            "gender_score": [0] * len(score),
            "truth_genders": truth_row.get("Gender", ""),
            "final_genders": final_row.get("gender", ""),
        }

    # Normal case: both have deities
    matched_deities = match_deities_with_aliases(deity_aliases, final_deities)
    score = []
    gender_score = []
    used_final_deities = set()

    for i, deity in enumerate(truth_deities):
        if deity is None:
            continue
        match_info = matched_deities.get(deity, {"match": "missing", "score": 0})
        matched_deity = match_info["match"]
        match_score = match_info["score"]

        found = match_score >= 50
        score.append(1 if found else 0)

        if found:
            used_final_deities.add(matched_deity)

        if found and i < len(truth_genders):
            truth_gender = truth_genders[i]
            try:
                final_idx = final_deities.index(matched_deity)
                final_gender = final_genders[final_idx] if final_idx < len(final_genders) else None
                gender_score.append(1 if truth_gender == final_gender else 0)
            except ValueError:
                gender_score.append(0)

    # Identify false positives — unmatched predicted deities
    # (These are already handled in evaluate_subset function)

    return {
        "uuid": uuid,
        "truth_deities": truth_row.get("Deities", ""),
        "final_deities": final_row.get("deities", ""),
        "matched_deities": str(matched_deities),
        "score": score,
        "gender_score": gender_score,
        "truth_genders": truth_row.get("Gender", ""),
        "final_genders": final_row.get("gender", ""),
    }


# ==============================================================================
# EVALUATION METRICS FUNCTIONS
# ==============================================================================


def evaluate_subset(results_df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate evaluation metrics (TP, FP, FN, precision, recall, F1, accuracy) for a subset.

    Args:
        results_df (pd.DataFrame): DataFrame with evaluation results from evaluate_row

    Returns:
        Dict[str, Any]: Dictionary containing all evaluation metrics
    """
    TP = 0
    FP = 0
    FN = 0
    matched_missing = 0
    total_truth_deities = 0
    total_gpt_deities = 0

    for _, row in results_df.iterrows():
        used_final_deities = set()

        # Parse matched_deities
        if isinstance(row["matched_deities"], str):
            matched_deities = ast.literal_eval(row["matched_deities"])
        else:
            matched_deities = row["matched_deities"]

        for deity, match_info in matched_deities.items():
            if isinstance(match_info, dict):
                match = match_info["match"]
                score = match_info["score"]
            else:
                match, score = match_info

            if deity.lower() == "missing" and match.lower() == "missing":
                matched_missing += 1
            elif score >= 50:
                TP += 1
                used_final_deities.add(match)
            else:
                if str(row["truth_deities"]).strip().lower() != "missing":
                    FN += 1

        final_deities = parse_comma_column(row["final_deities"])
        unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
        FP += len(unmatched_predictions)

        # Count raw totals
        truth_deities = parse_comma_column(row.get("truth_deities", ""))
        total_truth_deities += sum(1 for d in truth_deities if d)
        total_gpt_deities += sum(1 for d in final_deities if d)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Matched Missing": matched_missing,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Accuracy": round(accuracy, 2),
        "Truth_Deities": total_truth_deities,
        "GPT_Deities": total_gpt_deities,
    }


def evaluate_gender_matches_wide(truth_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate gender matching accuracy in wide format.

    For each gender category, calculates matched and unmatched counts
    (deity was identified correctly regardless of gender prediction).

    Args:
        truth_df (pd.DataFrame): Ground truth dataframe
        final_df (pd.DataFrame): Predicted dataframe

    Returns:
        Dict[str, Any]: Dictionary with gender matching statistics
    """
    results = []
    for _, truth_row in truth_df.iterrows():
        uuid = truth_row["uuid"]
        final_row = final_df[final_df["uuid"] == uuid]
        if final_row.empty:
            continue
        result = evaluate_row(uuid, truth_row, final_row.iloc[0])
        results.append(result)

    gender_match_stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})
    observed_genders = set()

    for row in results:
        truth_genders = parse_comma_column(row.get("truth_genders", ""))
        deity_scores = row.get("score", [])

        for gender, match in zip(truth_genders, deity_scores):
            if gender is None or str(gender).strip().lower() in {"", "missing"}:
                gender_key = "missing"
            else:
                gender_key = str(gender).strip().lower()
            observed_genders.add(gender_key)
            if match == 1:
                gender_match_stats[gender_key]["matched"] += 1
            else:
                gender_match_stats[gender_key]["unmatched"] += 1

    # Define genders to exclude
    genders_to_exclude = {"general", "mixed"}

    # Prioritized gender order
    priority = ["male", "female"]
    ordered_genders = priority + sorted(
        g for g in observed_genders if g not in priority and g not in genders_to_exclude
    )

    # Build dynamic summary row
    summary_row = {}
    for gender in ordered_genders:
        matched = gender_match_stats[gender]["matched"]
        unmatched = gender_match_stats[gender]["unmatched"]
        total = matched + unmatched
        pct = round(100 * matched / total, 2) if total > 0 else 0.0

        summary_row[f"m_{gender}"] = matched
        summary_row[f"u_{gender}"] = unmatched
        summary_row[f"t_{gender}"] = total
        summary_row[f"%_{gender}"] = pct

    return summary_row


def evaluate_gender_accuracy_wide(truth_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate gender prediction accuracy for matched deities in wide format.

    For deities that were correctly identified (match score = 1),
    checks if the gender prediction was also correct.

    Args:
        truth_df (pd.DataFrame): Ground truth dataframe
        final_df (pd.DataFrame): Predicted dataframe

    Returns:
        Dict[str, Any]: Dictionary with gender accuracy statistics
    """
    results = []
    for _, truth_row in truth_df.iterrows():
        uuid = truth_row["uuid"]
        final_row = final_df[final_df["uuid"] == uuid]
        if final_row.empty:
            continue
        result = evaluate_row(uuid, truth_row, final_row.iloc[0])
        results.append(result)

    # Gender accuracy stats: for matched deities, is gender prediction correct?
    gender_match_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    observed_genders = set()

    for row in results:
        truth_genders = parse_comma_column(row.get("truth_genders", ""))
        final_genders = parse_comma_column(row.get("final_genders", ""))
        deity_scores = row.get("score", [])

        # Only evaluate gender for successfully matched deities (score = 1)
        for truth_gender, final_gender, deity_score in zip(
            truth_genders, final_genders, deity_scores
        ):
            if deity_score == 1:  # Only for matched deities
                # Normalize gender values
                if truth_gender is None or str(truth_gender).strip().lower() in {"", "missing"}:
                    truth_gender_key = "missing"
                else:
                    truth_gender_key = str(truth_gender).strip().lower()

                if final_gender is None or str(final_gender).strip().lower() in {"", "missing"}:
                    final_gender_key = "missing"
                else:
                    final_gender_key = str(final_gender).strip().lower()

                observed_genders.add(truth_gender_key)

                # Check if the predicted gender matches the truth gender
                if truth_gender_key == final_gender_key:
                    gender_match_stats[truth_gender_key]["correct"] += 1
                else:
                    gender_match_stats[truth_gender_key]["incorrect"] += 1

    # Prioritized gender order
    priority = ["male", "female"]
    ordered_genders = priority + sorted(g for g in observed_genders if g not in priority)

    # Build dynamic summary row
    summary_row = {}
    for gender in ordered_genders:
        correct = gender_match_stats[gender]["correct"]
        incorrect = gender_match_stats[gender]["incorrect"]
        total = correct + incorrect
        pct = round(100 * correct / total, 2) if total > 0 else 0.0

        summary_row[f"m_{gender}"] = correct  # matched deities with correct gender
        summary_row[f"u_{gender}"] = incorrect  # matched deities with incorrect gender
        summary_row[f"t_{gender}"] = total  # total matched deities of this gender
        summary_row[f"%_{gender}"] = pct  # percentage with correct gender

    return summary_row


def calculate_fp_fn_tp(row: pd.Series) -> tuple[int, int, int]:
    """
    Calculate false positives, false negatives, and true positives for a single row.

    Args:
        row (pd.Series): Row containing matched_deities and final_deities

    Returns:
        Tuple[int, int, int]: (TP, FP, FN) counts
    """
    TP, FP, FN = 0, 0, 0
    used_final_deities = set()

    # Safely parse matched_deities if read from string
    if isinstance(row["matched_deities"], str):
        matched_deities = ast.literal_eval(row["matched_deities"])
    else:
        matched_deities = row["matched_deities"]

    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info["match"]
            score = match_info["score"]
        else:
            match, score = match_info  # fallback for older format

        if score >= 50:  # Threshold for TP
            TP += 1
            used_final_deities.add(match)
        else:
            FN += 1

    final_deities = parse_comma_column(row["final_deities"])
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    FP += len(unmatched_predictions)

    return TP, FP, FN


def count_cat_types(df: pd.DataFrame) -> tuple[int, int]:
    """
    Count total individual and multiple deity type entries across all rows.

    Args:
        df (pd.DataFrame): DataFrame with cat_type column

    Returns:
        Tuple[int, int]: (individual_count, multiple_count)
    """
    all_types = df["cat_type"].dropna().astype(str).str.lower().str.split(",")
    # Flatten and clean list
    flattened = [
        item.strip()
        for sublist in all_types
        for item in sublist
        if item.strip() and item.strip() not in {"na", "missing"}
    ]
    counts = Counter(flattened)
    return counts.get("individual", 0), counts.get("multiple", 0)


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================


def save_df_as_png(
    df: pd.DataFrame, filename: str, dpi: int = 200, fontsize: int = 10, row_height: float = 0.6
) -> None:
    """
    Save dataframe as PNG image.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Output filename
        dpi (int): Image resolution
        fontsize (int): Font size for table text
        row_height (float): Height per row in inches
    """
    nrows = len(df)

    fig_height = nrows * row_height + 1.5
    fig_width = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Automatically adjust column widths based on content
    try:
        table.auto_set_column_width(col=list(range(len(df.columns))))
    except AttributeError:
        print("auto_set_column_width requires matplotlib >= 3.6")

    table.scale(1.0, 1.2)
    plt.tight_layout(pad=1.0)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_category_comparison(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: str,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Plot category comparison between truth and prediction with error bars.

    Args:
        truth_df (pd.DataFrame): Ground truth dataframe
        pred_df (pd.DataFrame): Prediction dataframe
        output_path (str): Path to save the plot
        cat_cols (Optional[List[str]]): List of category columns to compare

    Returns:
        pd.DataFrame: Results dataframe with statistical tests
    """
    if cat_cols is None:
        cat_cols = [col for col in truth_df.columns if col.startswith("cat_") and col != "cat_type"]

    # Count category occurrences
    truth_counts = truth_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum().astype(int)
    pred_counts = pred_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum().astype(int)

    # Total deities (denominator for proportions)
    n_truth = truth_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).sum()
    n_pred = pred_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).sum()

    # DataFrame to store results
    results = []

    for cat in cat_cols:
        x1 = truth_counts[cat]
        x2 = pred_counts[cat]
        p1 = x1 / n_truth
        p2 = x2 / n_pred
        p_pool = (x1 + x2) / (n_truth + n_pred)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_truth + 1 / n_pred))

        # z-score and p-value
        z = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - norm.cdf(abs(z)))

        # 95% CI for p1 - p2
        ci_low = (p1 - p2) - 1.96 * se
        ci_high = (p1 - p2) + 1.96 * se

        # Mark significant differences
        sig = "*" if p_val < 0.05 else ""

        results.append(
            {
                "Category": cat,
                "Truth": x1,
                "Prediction": x2,
                "p1": p1,
                "p2": p2,
                "p-value": p_val,
                "CI lower": ci_low,
                "CI upper": ci_high,
                "Sig": sig,
            }
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values("Truth", ascending=False)

    # Plotting with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35

    # Bar plots
    bars1 = ax.bar(x - width / 2, results_df["Truth"], width, label="Truth", color="skyblue")
    bars2 = ax.bar(
        x + width / 2, results_df["Prediction"], width, label="Prediction", color="orange"
    )

    # Add error bars to Prediction bars
    ci_lowers = results_df["CI lower"]
    ci_uppers = results_df["CI upper"]
    error_lengths = (ci_uppers - ci_lowers) / 2
    scaled_error = error_lengths * n_pred

    ax.errorbar(
        x + width / 2,
        results_df["Prediction"],
        yerr=scaled_error,
        fmt="none",
        ecolor="gray",
        capsize=5,
        elinewidth=1.5,
    )

    # Add asterisks for significant differences
    for i, sig in enumerate(results_df["Sig"]):
        if sig == "*":
            y = max(results_df["Truth"].iloc[i], results_df["Prediction"].iloc[i]) + 5
            ax.text(x[i], y, "*", ha="center", va="bottom", fontsize=14, color="red")

    # Labels and formatting
    ax.set_title("Deities per Category: Ground Truth vs Prediction (w/ 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Category"], rotation=90)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    plt.close()

    return results_df


def plot_regional_metrics(
    match_df: pd.DataFrame,
    output_dir: str,
    region_col: str = "Region",
    subregion_col: str = "Subregion",
) -> None:
    """
    Plot FP/FN metrics per region and subregion.

    Creates multiple plots:
    - FP/FN percentages per region
    - FP/FN percentages per subregion
    - FP/FN counts per region
    - FP/FN counts per subregion
    - Combined total, FP, FN per region

    Args:
        match_df (pd.DataFrame): DataFrame with FP, FN columns and region information
        output_dir (str): Directory to save plots
        region_col (str): Name of region column
        subregion_col (str): Name of subregion column
    """
    import os

    # Count FP and FN per region
    fp_region = match_df.groupby(region_col)["FP"].sum()
    fn_region = match_df.groupby(region_col)["FN"].sum()
    region_counts = match_df.groupby(region_col).size()

    # Calculate FP and FN per Region
    fp_region = match_df.groupby("Region")["FP"].sum()
    fn_region = match_df.groupby("Region")["FN"].sum()
    tp_region = match_df.groupby("Region")["TP"].sum()

    # Calculate FP and FN rates properly
    fp_region_rate = (fp_region / (tp_region + fp_region)) * 100  # FP as % of all predictions
    fn_region_rate = (fn_region / (tp_region + fn_region)) * 100  # FN as % of all true positives
    # Count FP and FN per subregion
    fp_subregion = match_df[match_df["FP"] == 1].groupby(subregion_col).size()
    fn_subregion = match_df[match_df["FN"] == 1].groupby(subregion_col).size()
    subregion_counts = match_df.groupby(subregion_col).size()

    # Calculate percentages
    fp_subregion_percent = (fp_subregion / subregion_counts) * 100
    fn_subregion_percent = (fn_subregion / subregion_counts) * 100

    # Plot FP/FN percentage per region
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fp_region_rate.index, y=fp_region_rate.values)
    plt.title("False Positives per Region (Percentage)")
    plt.xlabel("Region")
    plt.ylabel("Percentage of False Positives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp_per_region_percentage.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fn_region_rate.index, y=fn_region_rate.values)
    plt.title("False Negatives per Region (Percentage)")
    plt.xlabel("Region")
    plt.ylabel("Percentage of False Negatives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fn_per_region_percentage.png"))
    plt.close()

    # Plot FP/FN percentage per subregion
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fp_subregion_percent.index, y=fp_subregion_percent.values)
    plt.title("False Positives per Subregion (Percentage)")
    plt.xlabel("Subregion")
    plt.ylabel("Percentage of False Positives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp_per_subregion_percentage.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fn_subregion_percent.index, y=fn_subregion_percent.values)
    plt.title("False Negatives per Subregion (Percentage)")
    plt.xlabel("Subregion")
    plt.ylabel("Percentage of False Negatives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fn_per_subregion_percentage.png"))
    plt.close()

    # Plot FP/FN counts per region
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fp_region.index, y=fp_region.values)
    plt.title("False Positives per Region")
    plt.xlabel("Region")
    plt.ylabel("Count of False Positives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp_per_region.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fn_region.index, y=fn_region.values)
    plt.title("False Negatives per Region")
    plt.xlabel("Region")
    plt.ylabel("Count of False Negatives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fn_per_region.png"))
    plt.close()

    # Plot FP/FN counts per subregion
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fp_subregion.index, y=fp_subregion.values)
    plt.title("False Positives per Subregion")
    plt.xlabel("Subregion")
    plt.ylabel("Count of False Positives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp_per_subregion.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fn_subregion.index, y=fn_subregion.values)
    plt.title("False Negatives per Subregion")
    plt.xlabel("Subregion")
    plt.ylabel("Count of False Negatives")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fn_per_subregion.png"))
    plt.close()

    # Combined plot: total, FP, FN per region
    total_region = match_df.groupby(region_col).size()

    region_summary = (
        pd.DataFrame(
            {
                "Total Deities": total_region,
                "False Positives": fp_region,
                "False Negatives": fn_region,
            }
        )
        .fillna(0)
        .astype(int)
    )

    # Reshape for plotting
    region_summary_long = region_summary.reset_index().melt(
        id_vars=region_col, var_name="Metric", value_name="Count"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=region_summary_long, x=region_col, y="Count", hue="Metric")
    plt.title("Total Deities, False Positives, and False Negatives per Region")
    plt.xlabel("Region")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp_fn_total_per_region.png"))
    plt.close()
