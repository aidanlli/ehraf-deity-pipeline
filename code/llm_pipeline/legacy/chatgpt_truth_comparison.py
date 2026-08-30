import ast
import math

import matplotlib.pyplot as plt
import pandas as pd
from fuzzywuzzy import fuzz, process

# File paths
truth_path = r"path/to/filtered_250_sampled_rows_v5.csv"
final_path = r"path/to/chatgpt_deity_output_7212025_cleaned.csv"

# Load datasets
truth_df = pd.read_csv(truth_path, encoding="utf-8-sig")
final_df = pd.read_csv(final_path, encoding="utf-8-sig")


# Normalize function
def normalize_name(name, keep_missing=False):
    if pd.isna(name):
        return None
    s = str(name).strip()
    s_lower = s.lower()

    if not s_lower:
        return None
    if s_lower in {"missing", "na"}:
        return s_lower if keep_missing else None

    return s_lower


# Normalize gender column in final_df
final_df["gender"] = (
    final_df["gender"].str.strip().str.lower().replace({"male & female": "general"})
)
# final_df['gender'] = final_df['gender'].str.strip().str.lower().replace({'general': 'missing'})


def save_df_as_png(df, filename, dpi=200, fontsize=10, row_height=0.6):
    nrows = len(df)

    # Rough estimate for initial figure width, will be corrected by auto_set_column_width
    fig_height = nrows * row_height + 1.5
    fig_width = 12  # Start with a moderate width

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


def parse_comma_column(column):
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


def parse_linked_column(column):
    return [normalize_name(part) for part in column.split(";")] if isinstance(column, str) else []


# Match deities with aliases
def match_deities_with_aliases(deity_aliases, final_deities):
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


# Evaluation function
def evaluate_row(uuid, truth_row, final_row):
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

    if all(d is None for d in truth_deities) and all(d is None for d in final_deities):
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get("Deities", ""),
            "final_deities": final_row.get("deities", ""),
            "matched_deities": str({"missing": ("missing", 100)}),
            "score": [1],
            "gender_score": [1],
        }

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
        }

    elif any(d is not None for d in truth_deities) and all(d is None for d in final_deities):
        score = [0 for d in truth_deities if d is not None]
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get("Deities", ""),
            "final_deities": final_row.get("deities", ""),
            "matched_deities": str({d: ("missing", 0) for d in truth_deities if d}),
            "score": score,
            "gender_score": [0] * len(score),
        }

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
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    score += [0 for _ in unmatched_predictions]  # Truth = 0, Predicted = 1
    gender_score += [0 for _ in unmatched_predictions]  # No gender match

    filtered_gender_score = [gs for ds, gs in zip(score, gender_score) if ds == 1]

    return {
        "uuid": uuid,
        "truth_deities": truth_row.get("Deities", ""),
        "truth_genders": truth_row.get("Gender", ""),
        "other_names": truth_row.get("Other_names", ""),
        "final_deities": final_row.get("deities", ""),
        "final_genders": final_row.get("gender", ""),
        "other_names_final": final_row.get("other names", ""),
        "matched_deities": str(matched_deities),
        "score": score,
        "gender_score": filtered_gender_score,
    }


# Run evaluation
results = []

for _, truth_row in truth_df.iterrows():
    uuid = truth_row["uuid"]
    final_row = final_df[final_df["uuid"] == uuid]
    if final_row.empty:
        continue
    result = evaluate_row(uuid, truth_row, final_row.iloc[0])
    results.append(result)
from collections import defaultdict

# Initialize counts
gender_match_stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})

for row in results:
    truth_genders = parse_comma_column(row.get("truth_genders", ""))
    deity_scores = row.get("score", [])

    for gender, match in zip(truth_genders, deity_scores):
        # Normalize gender key
        if gender is None or str(gender).strip().lower() in {"", "missing"}:
            gender_key = "missing"
        else:
            gender_key = str(gender).strip().lower()

        if match == 1:
            gender_match_stats[gender_key]["matched"] += 1
        else:
            gender_match_stats[gender_key]["unmatched"] += 1

# Convert to pandas DataFrame
gender_stats_df = pd.DataFrame(
    [
        {
            "Gender": gender,
            "Matched": counts["matched"],
            "Unmatched": counts["unmatched"],
            "Total": counts["matched"] + counts["unmatched"],
            "% Matched": round(
                100 * counts["matched"] / (counts["matched"] + counts["unmatched"]), 2
            )
            if (counts["matched"] + counts["unmatched"]) > 0
            else 0.0,
        }
        for gender, counts in gender_match_stats.items()
    ]
)
print(gender_stats_df)

# Optional: sort by total descending
gender_stats_df = gender_stats_df.sort_values(by="Total", ascending=False).reset_index(drop=True)

# Display the table
print(gender_stats_df)
save_df_as_png(gender_stats_df, r"path/to/gender_eval_summary.png")

# Convert results
results_df = pd.DataFrame(results)

TP, FP, FN = 0, 0, 0
gTP, gFP, gFN = 0, 0, 0
matched_missing = 0

for row in results:
    # Safely parse matched_deities if read from string
    if isinstance(row["matched_deities"], str):
        matched_deities = ast.literal_eval(row["matched_deities"])
    else:
        matched_deities = row["matched_deities"]

    used_final_deities = set()

    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info["match"]
            score = match_info["score"]
        else:
            match, score = match_info  # fallback for older format

        if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
            matched_missing += 1  # new counter
        elif score >= 50:
            TP += 1
            used_final_deities.add(match)
        else:
            if str(row["truth_deities"]).strip().lower() != "missing":
                FN += 1

    final_deities = parse_comma_column(row["final_deities"])
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    FP += len(unmatched_predictions)

    # Gender metrics
    gender_scores = row["gender_score"]
    deity_scores = row["score"]

    for ds, gs in zip(deity_scores, gender_scores):
        if ds == 1:
            if gs == 1:
                gTP += 1
            else:
                gFP += 1


# Compute metrics
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

g_precision = gTP / (gTP + gFP) if (gTP + gFP) else 0
g_recall = gTP / (gTP + gFN) if (gTP + gFN) else 0
g_f1 = 2 * g_precision * g_recall / (g_precision + g_recall) if (g_precision + g_recall) else 0
g_accuracy = gTP / (gTP + gFP + gFN) if (gTP + gFP + gFN) else 0


# Print results
print("Deity Matching:")
print(f"  TP: {TP}, FP: {FP}, FN: {FN}")
print(
    f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}\n"
)

print("Gender Matching:")
print(f"  TP: {gTP}, FP: {gFP}, FN: {gFN}")
print(
    f"  Precision: {g_precision:.2f}, Recall: {g_recall:.2f}, F1-Score: {g_f1:.2f}, Accuracy: {g_accuracy:.2f}"
)

print(f"  Matched Missing: {matched_missing}")
# Collect and save false positives and false negatives
false_positives = []
false_negatives = []
print(results_df.columns.tolist())

for row in results:
    if isinstance(row["matched_deities"], str):
        matched_deities = ast.literal_eval(row["matched_deities"])
    else:
        matched_deities = row["matched_deities"]

    used_final_deities = set()
    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info["match"]
            score = match_info["score"]
        else:
            match, score = match_info

        if score >= 50:
            used_final_deities.add(match)
        else:
            if str(row["truth_deities"]).strip().lower() == "missing":
                continue  # Skip FN if truth_deities is 'missing'
            # False Negative
            false_negatives.append(
                {
                    "uuid": row["uuid"],
                    "missing_deity": deity,
                    "truth_deities": row["truth_deities"],
                    "predicted_deities": row["final_deities"],
                    "matched_deities": row["matched_deities"],
                }
            )

    final_deities = parse_comma_column(row["final_deities"])

    gender_data = row.get("final_genders", [])
    if isinstance(gender_data, str):
        gender_str = gender_data
    elif isinstance(gender_data, float) and math.isnan(gender_data):
        gender_str = ""
    elif isinstance(gender_data, list):
        gender_str = ", ".join(map(str, gender_data))
    else:
        gender_str = str(gender_data)
    gender_data_truth = row.get("truth_genders", [])

    if isinstance(gender_data_truth, str):
        gender_str_truth = gender_data_truth
    elif isinstance(gender_data_truth, float) and math.isnan(gender_data_truth):
        gender_str_truth = ""
    elif isinstance(gender_data_truth, list):
        gender_str_truth = ", ".join(map(str, gender_data_truth))
    else:
        gender_str_truth = str(gender_data_truth)

    for d in final_deities:
        if d not in used_final_deities:
            false_positives.append(
                {
                    "uuid": row["uuid"],
                    "extra_deity": d,
                    "truth_deities": row["truth_deities"],
                    "truth_gender": gender_str_truth,
                    "predicted_deities": row["final_deities"],
                    "predicted gender": gender_str,  # now guaranteed to be a list
                    "matched_deities": row["matched_deities"],
                }
            )


# Save FP and FN to CSV
fp_df = pd.DataFrame(false_positives)
fn_df = pd.DataFrame(false_negatives)

fp_df.to_csv(r"path/to/deity_false_positives_v4.csv", index=False, encoding="utf-8-sig")
fn_df.to_csv(r"path/to/deity_false_negatives_v4.csv", index=False, encoding="utf-8-sig")

# Save
results_df = results_df.merge(truth_df[["uuid", "ambiguous"]], on="uuid", how="left")
results_df.to_csv(
    r"path/to/deity_matching_results_full_v8_test.csv", index=False, encoding="utf-8-sig"
)

import ast

import pandas as pd


def evaluate_deities(truth_df, final_df):
    # alias_map = build_alias_map(truth_df)
    # final_df = clean_final_deities_into_aliases(final_df, truth_df)

    results = []

    total_truth_deities = 0
    total_gpt_deities = 0

    for _, truth_row in truth_df.iterrows():
        uuid = truth_row["uuid"]
        final_row = final_df[final_df["uuid"] == uuid]
        if final_row.empty:
            continue
        result = evaluate_row(uuid, truth_row, final_row.iloc[0])
        results.append(result)

        # Count deities from truth and final
        truth_deities = parse_comma_column(truth_row.get("Deities", ""))
        final_deities = parse_comma_column(final_row.iloc[0].get("deities", ""))

        total_truth_deities += sum(1 for d in truth_deities if d)
        normalized_deities = set(normalize_name(d) for d in final_deities if d)
        total_gpt_deities += len(normalized_deities)

    TP, FP, FN = 0, 0, 0
    matched_missing = 0
    for row in results:
        matched_deities = (
            ast.literal_eval(row["matched_deities"])
            if isinstance(row["matched_deities"], str)
            else row["matched_deities"]
        )
        used_final_deities = set()

        for deity, match_info in matched_deities.items():
            if isinstance(match_info, dict):
                match = match_info["match"]
                score = match_info["score"]
            else:
                match, score = match_info

            if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
                matched_missing += 1  # new counter
            elif score >= 50:
                TP += 1
                used_final_deities.add(match)
            else:
                if str(row["truth_deities"]).strip().lower() != "missing":
                    FN += 1

        final_deities = parse_comma_column(row["final_deities"])
        unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
        FP += len(unmatched_predictions)

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


# Load all combinations
truth = pd.read_csv(r"path/to/filtered_250_sampled_rows.csv")
truth_v2 = pd.read_csv(r"path/to/filtered_250_sampled_rows_v2.csv")
truth_v3 = pd.read_csv(r"path/to/filtered_250_sampled_rows_v3.csv")
truth_v4 = pd.read_csv(r"path/to/filtered_250_sampled_rows_v4.csv")
truth_v5 = pd.read_csv(r"path/to/filtered_250_sampled_rows_v5.csv")
truth_v6 = pd.read_csv(r"path/to/filtered_250_sampled_rows_v6.csv")
final = pd.read_csv(r"path/to/3.final_full_cleaned.csv")
final_v2 = pd.read_csv(r"path/to/3.final_full_cleaned_v2.csv")
final_v3 = pd.read_csv(r"path/to/3.final_full_cleaned_v3.csv")
final_v4 = pd.read_csv(r"path/to/3.final_full_cleaned_v4.csv")
final_v5 = pd.read_csv(r"path/to/3.chatgpt_deity_output_792025_cleaned.csv")
final_v6 = pd.read_csv(r"path/to/3.chatgpt_deity_output_792025_v2_cleaned.csv")
final_v7 = pd.read_csv(r"path/to/chatgpt_deity_output_7212025_cleaned.csv")
final_gemini_v1 = pd.read_csv(r"path/to/3.gemini_api_test_07212025_cleaned.csv")


# Prepare comparison matrix
combinations = [
    ("Truth 1 with Prompt 1", truth, final),
    ("Truth 1 vs Prompt 2", truth, final_v2),
    # ("Truth 2 vs Prompt 1", truth_v2, final),
    # ("Truth 2 vs Prompt 2", truth_v2, final_v2),
    ("Truth 3 vs Prompt 2", truth_v3, final_v2),
    ("Truth 4 vs Prompt 3", truth_v4, final_v3),
    ("Truth 5 vs Prompt 4", truth_v5, final_v4),
    ("Truth 5 vs Prompt 3 Updated GPT", truth_v5, final_v5),
    ("Truth 5 vs Prompt 5 Updated GPT", truth_v5, final_v6),
    ("Truth 6 vs Prompt 6 Updated GPT", truth_v6, final_v7),
    ("Truth 6 vs Gemini v1", truth_v6, final_gemini_v1),
]

results_summary = []

for name, t_df, f_df in combinations:
    metrics = evaluate_deities(t_df, f_df)
    metrics["Comparison"] = name
    results_summary.append(metrics)

# Create and display the summary table
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df[
    [
        "Comparison",
        "TP",
        "FP",
        "FN",
        "Matched Missing",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Truth_Deities",
        "GPT_Deities",
    ]
]
print(summary_df)

# Save with improved layout
save_df_as_png(summary_df, r"path/to/deity_eval_summary.png")


ambiguous_1_df = results_df[results_df["ambiguous"] == 1]
ambiguous_0_df = results_df[results_df["ambiguous"] == 0]


def evaluate_subset(results_subset):
    TP, FP, FN = 0, 0, 0
    matched_missing = 0
    total_truth_deities = 0
    total_gpt_deities = 0

    for _, row in results_subset.iterrows():
        matched_deities = (
            ast.literal_eval(row["matched_deities"])
            if isinstance(row["matched_deities"], str)
            else row["matched_deities"]
        )
        used_final_deities = set()

        for deity, match_info in matched_deities.items():
            if isinstance(match_info, dict):
                match = match_info["match"]
                score = match_info["score"]
            else:
                match, score = match_info

            if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
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


# Evaluate both groups
amb_1_metrics = evaluate_subset(ambiguous_1_df)
amb_1_metrics["Comparison"] = "Ambiguous = 1"

amb_0_metrics = evaluate_subset(ambiguous_0_df)
amb_0_metrics["Comparison"] = "Ambiguous = 0"

# Combine and display
amb_summary_df = pd.DataFrame([amb_0_metrics, amb_1_metrics])
amb_summary_df = amb_summary_df[
    [
        "Comparison",
        "TP",
        "FP",
        "FN",
        "Matched Missing",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Truth_Deities",
        "GPT_Deities",
    ]
]
print(amb_summary_df)

# Optional: save
save_df_as_png(amb_summary_df, r"path/to/deity_eval_ambiguous_summary.png")

from collections import defaultdict


def parse_comma_column(column):
    if isinstance(column, str):
        stripped = column.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                evaluated = ast.literal_eval(stripped)
                if isinstance(evaluated, list):
                    return [normalize_name(item) for item in evaluated]
            except (ValueError, SyntaxError):
                pass

        # Return normalized (including Nones) without filtering them out
        return [normalize_name(part) for part in column.split(",")]
    return []


def evaluate_gender_matches_wide(truth_df, final_df):
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

    # Prioritized gender order, now with exclusion
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


gender_wide_summary = []

for name, t_df, f_df in combinations:
    row = evaluate_gender_matches_wide(t_df, f_df)
    row["Comparison"] = name
    gender_wide_summary.append(row)

gender_wide_df = pd.DataFrame(gender_wide_summary)

# Optional: move "Comparison" to the first column
cols = ["Comparison"] + [col for col in gender_wide_df.columns if col != "Comparison"]
gender_wide_df = gender_wide_df[cols]

print(gender_wide_df)
save_df_as_png(gender_wide_df, r"path/to/gender_eval_summary_wide.png")


def evaluate_gender_accuracy_wide(truth_df, final_df):
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


# Loop across comparisons
gender_wide_summary = []

for name, t_df, f_df in combinations:
    row = evaluate_gender_accuracy_wide(t_df, f_df)
    row["Comparison"] = name
    gender_wide_summary.append(row)

gender_wide_df = pd.DataFrame(gender_wide_summary)
cols = ["Comparison"] + [col for col in gender_wide_df.columns if col != "Comparison"]
gender_wide_df = gender_wide_df[cols]

print(gender_wide_df)
save_df_as_png(gender_wide_df, r"path/to/gender_eval_summary_accuracy.png")
