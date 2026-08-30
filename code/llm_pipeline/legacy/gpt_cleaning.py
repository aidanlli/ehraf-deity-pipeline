import ast

import pandas as pd

input_path = "path/to/chatgpt_deity_output_7212025.csv"
output_path = "path/to/chatgpt_deity_output_7212025_cleaned.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")

# Columns to preserve and format
list_format_columns = ["deities", "gender", "other names"]


def standardize_list_cell(cell, column_name):
    if pd.isna(cell):
        return ""
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            cleaned_items = []
            for item in parsed:
                item_str = str(item).strip().strip('"')
                if column_name == "gender":
                    cleaned_items.append(item_str)
                else:
                    if item_str.lower() == "missing":
                        cleaned_items.append(item_str)
                    else:
                        cleaned_items.append(f'"{item_str}"')
            return ", ".join(cleaned_items)
    except:
        pass

    # If not a list
    cell_str = str(cell).strip().strip('"')
    if column_name == "gender" or cell_str.lower() == "missing":
        return cell_str
    else:
        return f'"{cell_str}"'


def clean_other_cell(cell):
    if pd.isna(cell):
        return ""
    return str(cell).replace("[", "").replace("]", "").replace("'", "")


# Apply appropriate cleaning to each column
for col in df.columns:
    if col in list_format_columns:
        df[col] = df[col].apply(lambda x: standardize_list_cell(x, col))
    else:
        df[col] = df[col].apply(clean_other_cell)

# Save cleaned data
df.to_csv(output_path, index=False, encoding="utf-8-sig")
