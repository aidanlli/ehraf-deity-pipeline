import ast
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# === Load Data ===
filtered_df = pd.read_csv("path/to/filtered_250_sampled_rows_v2.csv")
full_df = pd.read_csv("path/to/3.final_full_cleaned.csv")
match_df = pd.read_csv("path/to/deity_matching_results_full_v2.csv")


def normalize_name(name):
    if pd.isna(name) or str(name).strip().lower() in {"", "missing"}:
        return None
    return str(name).strip().lower()


def parse_comma_column(column):
    if isinstance(column, str):
        # Heuristic: looks like a list if it starts with [ and ends with ]
        if column.strip().startswith("[") and column.strip().endswith("]"):
            try:
                evaluated = ast.literal_eval(column)
                if isinstance(evaluated, list):
                    return [normalize_name(item) for item in evaluated]
            except (ValueError, SyntaxError):
                pass  # Fall through to comma split

        # Fallback: treat as normal text, split by commas
        return [normalize_name(part) for part in column.split(",")]
    return []


# === Clean 'matched_deities' column ===
def extract_matched(d):
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


valid_filtered_df = filtered_df[
    ~filtered_df["Deities"].fillna("").str.strip().str.lower().eq("missing")
].copy()

# Apply extract_matched to valid rows only
valid_filtered_df["matched_list"] = valid_filtered_df["Deities"].apply(extract_matched)

# === 1. Number of deities identified in both datasets ===
valid_filtered_df["truth_list"] = (
    valid_filtered_df["Deities"]
    .fillna("")
    .apply(lambda x: [i.strip().strip('"') for i in x.split(",") if i.strip()])
)
valid_filtered_df["num_truth"] = valid_filtered_df["truth_list"].apply(len)
valid_filtered_df["num_matched"] = valid_filtered_df["matched_list"].apply(len)

# Parse full_df deities as before
full_df["parsed_deities"] = full_df["deities"].dropna().apply(parse_comma_column)

# === Totals (excluding "missing" deities from filtered_df only) ===
num_deities_total = valid_filtered_df["num_truth"].sum()
num_deities_matched = valid_filtered_df["num_matched"].sum()
total_deities_in_full_df = (
    full_df["parsed_deities"].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
)


print(f"Total deities in gpt output: {total_deities_in_full_df}")
print(f"Total deities in ground truth: {num_deities_total}")
print(f"Total matched deities: {num_deities_matched}")


# === 2. Number of deities per gender ===
def expand_gender(row):
    genders = [g.strip() for g in str(row).split(",")]
    return pd.Series(genders)


gender_expanded = match_df["truth_genders"].dropna().apply(lambda x: x.split(","))
gender_list = [g.strip().lower() for sub in gender_expanded for g in sub]
gender_counts = pd.Series(gender_list).value_counts()
print("\nDeities per gender:\n", gender_counts)

# === 3. False Positives / False Negatives per gender ===
match_df["FN"] = match_df["score"].apply(lambda x: 1 if 0 in ast.literal_eval(x) else 0)
match_df["FP"] = match_df["score"].apply(
    lambda x: 1 if len(ast.literal_eval(x)) > sum(ast.literal_eval(x)) else 0
)

# We'll break this down by `truth_genders` for FN and `final_genders` for FP
fn_gender = (
    match_df[match_df["FN"] == 1]["truth_genders"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)
fp_gender = (
    match_df[match_df["FP"] == 1]["final_genders"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)

print("\nFalse Negatives per gender:\n", fn_gender)
print("\nFalse Positives per gender:\n", fp_gender)
downloads_folder = "path/to/"

# === 4. Deities per category type (Side-by-side comparison between full_df and filtered_df) ===
# Count categories in full_df
# Category columns
cat_cols = [col for col in full_df.columns if col.startswith("cat_") and col != "cat_type"]

# Count category occurrences
truth_counts = filtered_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum().astype(int)
pred_counts = full_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum().astype(int)

# Total deities (denominator for proportions)
n_truth = filtered_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).sum()
n_pred = full_df[cat_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).sum()

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

# === Plotting with error bars ===
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

# Bar plots
bars1 = ax.bar(x - width / 2, results_df["Truth"], width, label="Truth", color="skyblue")
bars2 = ax.bar(x + width / 2, results_df["Prediction"], width, label="Prediction", color="orange")

# Add error bars to Prediction bars (CI on difference, centered at prediction)
ci_lowers = results_df["CI lower"]
ci_uppers = results_df["CI upper"]
error_lengths = (ci_uppers - ci_lowers) / 2  # symmetric error bars
scaled_error = error_lengths * n_pred
# Error bars
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
plt.savefig(os.path.join(downloads_folder, "deities_per_category_error.png"))


# === 5. Deities per classification (cat_type) ===


# Function to count total individual and multiple entries across all rows
def count_cat_types(df):
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


# Count in full_df and filtered_df
cat_type_counts_individual_full, cat_type_counts_multiple_full = count_cat_types(full_df)
cat_type_counts_individual_filtered, cat_type_counts_multiple_filtered = count_cat_types(
    filtered_df
)
print("Total rows in filtered_df:", len(filtered_df))
print("Rows with valid cat_type:", filtered_df["cat_type"].notna().sum())
print(
    "Rows with actual 'individual' or 'multiple':",
    filtered_df["cat_type"].str.lower().str.contains("individual|multiple", na=False).sum(),
)

# Print cat_type counts (individual and multiple) for both datasets
print("\nDeities per cat_type (individual and multiple) - Full Dataset:")
print(f"Individual: {cat_type_counts_individual_full}")
print(f"Multiple: {cat_type_counts_multiple_full}")

print("\nDeities per cat_type (individual and multiple) - Filtered Dataset:")
print(f"Individual: {cat_type_counts_individual_filtered}")
print(f"Multiple: {cat_type_counts_multiple_filtered}")

# === Plotting the comparison between full_df and filtered_df ===
cat_type_comparison = pd.DataFrame(
    {
        "Full Dataset": [cat_type_counts_individual_full, cat_type_counts_multiple_full],
        "Filtered Dataset": [
            cat_type_counts_individual_filtered,
            cat_type_counts_multiple_filtered,
        ],
    },
    index=["Individual", "Multiple"],
)

# Plotting side-by-side bar charts for the comparison
cat_type_comparison.plot(kind="bar", figsize=(8, 6), width=0.8)
plt.title("Deities per Classification (Individual vs Multiple) - Full vs Filtered Dataset")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot in the Downloads folder
downloads_folder = "path/to/"
plt.savefig(os.path.join(downloads_folder, "deities_per_classification_comparison.png"))

# === 6. Correlation between score and certainty_deity (position-matched, un-averaged) ===

# Merge certainty_deity into match_df
match_df = match_df.merge(full_df[["uuid", "certainty_deity"]], on="uuid", how="left")

# Extract certainty-score pairs
certainty_score_pairs = []

# Parse score strings to lists
match_df["score_list"] = match_df["score"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

for _, row in match_df.iterrows():
    certs = row.get("certainty_deity")
    scores = row.get("score_list")

    if not isinstance(scores, list) or not isinstance(certs, str):
        continue

    try:
        certainty_list = [float(x.strip()) for x in certs.split(",") if x.strip()]
        if len(certainty_list) != len(scores):
            continue
        certainty_score_pairs.extend(zip(certainty_list, scores))
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        continue

# Check and compute correlation
if not certainty_score_pairs:
    print("No valid certainty-score pairs found.")
else:
    certainties, scores = zip(*certainty_score_pairs)
    correlation = np.corrcoef(certainties, scores)[0, 1]
    print(f"\nCorrelation between match score and certainty_deity: {correlation:.3f}")

    # Count frequencies of each (certainty, score) pair
    pair_counts = Counter(zip(certainties, scores))

    # Prepare plot data
    x_vals, y_vals, sizes = [], [], []
    for (x, y), count in pair_counts.items():
        x_vals.append(x)
        y_vals.append(y)
        sizes.append(count * 40)  # Adjust multiplier for visibility

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6, edgecolor="black")

    # Annotate counts
    for x, y, count in zip(x_vals, y_vals, pair_counts.values()):
        if count > 1:
            plt.text(x, y, str(count), fontsize=9, ha="center", va="center")

    # Labels and title
    plt.title("Certainty vs. Match Score (Point Size = Frequency)")
    plt.xlabel("Certainty (Deity)")
    plt.ylabel("Match Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(downloads_folder, "certainty_vs_score_size_scaled.png"))

# === 7. FP/FN per Region, Subregion ===
# Merge uuid info back in
meta_cols = ["uuid", "Region", "Subregion"]
meta_df = filtered_df[meta_cols].drop_duplicates()

match_df = match_df.merge(meta_df, on="uuid", how="left")


# === TP, FP, FN Calculation per Row ===
def calculate_fp_fn_tp(row):
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


# === Apply TP, FP, FN Calculation ===
match_df[["TP", "FP", "FN"]] = match_df.apply(calculate_fp_fn_tp, axis=1, result_type="expand")

# === Count FP and FN per Region ===
fp_region = match_df.groupby("Region")["FP"].sum()
fn_region = match_df.groupby("Region")["FN"].sum()
region_counts = match_df.groupby("Region").size()

# Calculate FP and FN percentages per region
fp_region_percent = (fp_region / region_counts) * 100
fn_region_percent = (fn_region / region_counts) * 100

# Count FP and FN per subregion
fp_subregion = match_df[match_df["FP"] == 1].groupby("Subregion").size()
fn_subregion = match_df[match_df["FN"] == 1].groupby("Subregion").size()

# Count total entries per subregion (for percentage calculation)
subregion_counts = match_df.groupby("Subregion").size()

# Calculate FP and FN percentages per subregion
fp_subregion_percent = (fp_subregion / subregion_counts) * 100
fn_subregion_percent = (fn_subregion / subregion_counts) * 100

# Save FP/FN per region and subregion percentage figures in Downloads folder
plt.figure(figsize=(10, 6))
sns.barplot(x=fp_region_percent.index, y=fp_region_percent.values)
plt.title("False Positives per Region (Percentage)")
plt.xlabel("Region")
plt.ylabel("Percentage of False Positives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(downloads_folder, "fp_per_region_percentage.png"))

plt.figure(figsize=(10, 6))
sns.barplot(x=fn_region_percent.index, y=fn_region_percent.values)
plt.title("False Negatives per Region (Percentage)")
plt.xlabel("Region")
plt.ylabel("Percentage of False Negatives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(downloads_folder, "fn_per_region_percentage.png"))

plt.figure(figsize=(10, 6))
sns.barplot(x=fp_subregion_percent.index, y=fp_subregion_percent.values)
plt.title("False Positives per Subregion (Percentage)")
plt.xlabel("Subregion")
plt.ylabel("Percentage of False Positives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(downloads_folder, "fp_per_subregion_percentage.png"))

plt.figure(figsize=(10, 6))
sns.barplot(x=fn_subregion_percent.index, y=fn_subregion_percent.values)
plt.title("False Negatives per Subregion (Percentage)")
plt.xlabel("Subregion")
plt.ylabel("Percentage of False Negatives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(downloads_folder, "fn_per_subregion_percentage.png"))

# === FP/FN per Region ===
# Plotting FP counts per Region
plt.figure(figsize=(10, 6))
sns.barplot(x=fp_region.index, y=fp_region.values)
plt.title("False Positives per Region")
plt.xlabel("Region")
plt.ylabel("Count of False Positives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("path/to/fp_per_region.png")

# Plotting FN counts per Region
plt.figure(figsize=(10, 6))
sns.barplot(x=fn_region.index, y=fn_region.values)
plt.title("False Negatives per Region")
plt.xlabel("Region")
plt.ylabel("Count of False Negatives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("path/to/fn_per_region.png")

# === FP/FN per Subregion ===
# Plotting FP counts per Subregion
plt.figure(figsize=(10, 6))
sns.barplot(x=fp_subregion.index, y=fp_subregion.values)
plt.title("False Positives per Subregion")
plt.xlabel("Subregion")
plt.ylabel("Count of False Positives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("path/to/fp_per_subregion.png")

# Plotting FN counts per Subregion
plt.figure(figsize=(10, 6))
sns.barplot(x=fn_subregion.index, y=fn_subregion.values)
plt.title("False Negatives per Subregion")
plt.xlabel("Subregion")
plt.ylabel("Count of False Negatives")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("path/to/fn_per_subregion.png")

# Print out false positives and false negatives per region/subregion
print("\nFalse Positives per Region (Percentage):\n", fp_region_percent)
print("\nFalse Negatives per Region (Percentage):\n", fn_region_percent)

print("\nFalse Positives per Subregion (Percentage):\n", fp_subregion_percent)
print("\nFalse Negatives per Subregion (Percentage):\n", fn_subregion_percent)


# === Combine total, FP, FN per region into a single DataFrame ===
total_region = match_df.groupby("Region").size()

region_summary = (
    pd.DataFrame(
        {"Total Deities": total_region, "False Positives": fp_region, "False Negatives": fn_region}
    )
    .fillna(0)
    .astype(int)
)

# Reshape for plotting (long-form)
region_summary_long = region_summary.reset_index().melt(
    id_vars="Region", var_name="Metric", value_name="Count"
)

# === Plot grouped bar chart ===
plt.figure(figsize=(12, 6))
sns.barplot(data=region_summary_long, x="Region", y="Count", hue="Metric")
plt.title("Total Deities, False Positives, and False Negatives per Region")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(downloads_folder, "fp_fn_total_per_region.png"))
