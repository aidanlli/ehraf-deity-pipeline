"""Generate summary tables and plots describing the scraped corpus.

Step 06 of the scraping pipeline. Produces counts and charts for regions,
cultures, document types, subsistence methods, and gendered keyword
frequencies. Figures are written to ``output/figures/dataset_summary/``.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FIGURES_SUMMARY, PARAGRAPH_TEXT_CSV

input_path = PARAGRAPH_TEXT_CSV
export_dir = f"{FIGURES_SUMMARY}/"  # Ensure trailing slash

df = pd.read_csv(input_path)

# Define keywords
male_keywords = [
    "he",
    "him",
    "his",
    "father",
    "king",
    "lord",
    "prince",
    "son",
    "husband",
    "brother",
    "patriarch",
    "fatherhood",
    "brotherhood",
]
female_keywords = [
    "she",
    "her",
    "mother",
    "queen",
    "lady",
    "goddess",
    "daughter",
    "wife",
    "sister",
    "matriarch",
    "motherhood",
]
dual_keywords = ["androgynous", "dual-gendered", "god/goddess"]


# Function to count keywords
def count_keywords(column, keywords):
    counts = {}
    total = 0
    for keyword in keywords:
        count = column.str.count(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE).sum()
        counts[keyword] = count
        total += count
    return counts, total


# Count keywords
male_counts, male_total = count_keywords(df["Text"], male_keywords)
female_counts, female_total = count_keywords(df["Text"], female_keywords)
dual_counts, dual_total = count_keywords(df["Text"], dual_keywords)

# Convert keyword counts to DataFrames
male_df = pd.DataFrame(list(male_counts.items()), columns=["Male Keyword", "Count"])
male_df.loc[len(male_df)] = ["Total Male Keywords", male_total]

female_df = pd.DataFrame(list(female_counts.items()), columns=["Female Keyword", "Count"])
female_df.loc[len(female_df)] = ["Total Female Keywords", female_total]

dual_df = pd.DataFrame(list(dual_counts.items()), columns=["Dual Keyword", "Count"])
dual_df.loc[len(dual_df)] = ["Total Dual Keywords", dual_total]


# Function to save a DataFrame as an image
def save_df_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.5 + 1))  # Adjust figure size based on rows
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.savefig(f"{export_dir}{filename}", bbox_inches="tight", dpi=300)
    plt.close()


# Function to save a bar chart
def save_bar_chart(df, filename, title, x_label, y_label):
    plt.figure(figsize=(10, 5))
    plt.bar(df.iloc[:-1, 0], df.iloc[:-1, 1], color="skyblue")  # Exclude the total row
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{export_dir}{filename}", dpi=300)
    plt.close()


# Save keyword tables and charts
save_df_as_image(male_df, "male_keywords.png", "Male Keywords Count")
save_bar_chart(male_df, "male_keywords_bar.png", "Male Keywords Frequency", "Keyword", "Count")

save_df_as_image(female_df, "female_keywords.png", "Female Keywords Count")
save_bar_chart(
    female_df, "female_keywords_bar.png", "Female Keywords Frequency", "Keyword", "Count"
)

save_df_as_image(dual_df, "dual_keywords.png", "Dual Keywords Count")
save_bar_chart(dual_df, "dual_keywords_bar.png", "Dual Keywords Frequency", "Keyword", "Count")

# Count unique values for categorical columns
columns_to_count = ["DocType", "Culture", "Region", "Subregion", "Subsistence"]

for column in columns_to_count:
    count_df = df[column].value_counts().reset_index()
    count_df.columns = [column, "Count"]
    total_row = pd.DataFrame({column: [f"Total {column}"], "Count": [count_df["Count"].sum()]})
    count_df = pd.concat([count_df, total_row], ignore_index=True)
    if column == "Culture":
        count_df.to_csv(f"{export_dir}{column}_counts.csv", index=False)
    else:
        save_df_as_image(count_df, f"{column}_counts.png", f"{column} Value Counts")
        save_bar_chart(
            count_df.iloc[:-1], f"{column}_counts_bar.png", f"{column} Frequency", column, "Count"
        )  # Exclude total row


# Original Culture histogram
plt.figure(figsize=(10, 5))
df["Culture"].value_counts().plot(kind="hist", bins=50, color="lightcoral", edgecolor="black")
plt.xlabel("Culture")
plt.ylabel("Number of Paragraphs")
plt.title("Culture Count Distribution")
plt.tight_layout()
plt.savefig(f"{export_dir}Culture_histogram.png", dpi=300)
plt.close()

max_bin = 3000  # Upper x-limit
num_bins = 60  # Number of bins
bin_edges = np.linspace(0, max_bin, num_bins + 1)  # Creates evenly spaced bin edges

# Additional Culture histogram with x-axis limited to 0-3000
plt.figure(figsize=(10, 5))
df["Culture"].value_counts().plot(
    kind="hist", bins=bin_edges, color="lightcoral", edgecolor="black"
)
plt.xlabel("Culture")
plt.ylabel("Number of Paragraphs")
plt.title("Culture Count Distribution")
plt.xlim(0, 3000)  # Set x-axis limits
plt.tight_layout()
plt.savefig(f"{export_dir}Culture_histogram_limited.png", dpi=300)
plt.close()

# Count the number of unique cultures per region
unique_cultures_per_region = df.groupby("Region")["Culture"].nunique().reset_index()
unique_cultures_per_region.columns = ["Region", "Unique Culture Count"]

# Save as a table image
save_df_as_image(
    unique_cultures_per_region, "unique_cultures_per_region.png", "Unique Cultures per Region"
)

# Optional: Save as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(
    unique_cultures_per_region["Region"],
    unique_cultures_per_region["Unique Culture Count"],
    color="mediumseagreen",
)
plt.xlabel("Region")
plt.ylabel("Number of Unique Cultures")
plt.title("Unique Cultures per Region")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{export_dir}unique_cultures_per_region_bar.png", dpi=300)
plt.close()

# Count DocType occurrences per Region and save as CSV
doc_type_by_region = df.groupby(["Region", "DocType"]).size().reset_index(name="Count")
doc_type_by_region.to_csv(f"{export_dir}doc_type_by_region.csv", index=False)


# Pivot the data for better visualization
doc_type_pivot = doc_type_by_region.pivot(index="Region", columns="DocType", values="Count").fillna(
    0
)

# Plot the stacked bar chart
plt.figure(figsize=(12, 6))
doc_type_pivot.plot(kind="bar", stacked=True, colormap="tab10", figsize=(12, 6))

plt.xlabel("Region")
plt.ylabel("Count")
plt.title("DocType Counts by Region")
plt.xticks(rotation=45, ha="right")
plt.legend(
    title="DocType", bbox_to_anchor=(1.05, 1), loc="upper left"
)  # Move legend outside the plot
plt.tight_layout()
plt.savefig(f"{export_dir}doc_type_by_region_bar.png", dpi=300)
plt.close()


# Get unique regions
regions = doc_type_by_region["Region"].unique()
num_regions = len(regions)

# Set up grid layout (e.g., 3x3, 4x4 depending on region count)
cols = 4  # Adjust for aesthetics
rows = math.ceil(num_regions / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 9))
axes = axes.flatten()  # Flatten to 1D array for easy iteration

# Generate pie charts for each region
for i, region in enumerate(regions):
    subset = doc_type_by_region[doc_type_by_region["Region"] == region]

    # Only plot if there's more than one category
    if len(subset) > 1:
        axes[i].pie(
            subset["Count"],
            labels=subset["DocType"],
            autopct="%1.1f%%",
            colors=plt.cm.Paired.colors,
        )
        axes[i].set_title(region)
    else:
        axes[i].axis("off")  # Hide empty subplots

# Adjust layout and save
plt.tight_layout()
plt.savefig(f"{export_dir}doc_type_pie_all_regions.png", dpi=300)
plt.close()


print("Tables and charts saved as PNG images in:", export_dir)
