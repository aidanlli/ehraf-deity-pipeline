"""Alternative single-prompt identification pipeline using Google Gemini.

Replicates the GPT-4o pipeline with one combined identification +
classification prompt against gemini-2.5-flash. Requires a Google API key
in the environment (see .env.example). The input sample must include the
eHRAF "Text" column, which is licensed and therefore not shipped in this
repository -- regenerate it with the scraping pipeline (see README).
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_INTERMEDIATE

# === SETUP ===
# API key configured via environment variable
client = genai.Client()

# Sample with regenerated "Text" column (see module docstring)
input_csv_path = DATA_INTERMEDIATE / "250_sampled_rows_with_text.csv"
output_csv_path = DATA_INTERMEDIATE / "gemini_results.csv"
batch_size = 2
temperature = 0

# === LOAD DATA ===
df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
if os.path.exists(output_csv_path):
    result_df = pd.read_csv(output_csv_path, encoding="utf-8-sig")
    last_processed_index = result_df.index[-1]
else:
    result_df = pd.DataFrame()
    last_processed_index = -1


# === JSON PARSER ===
def extract_structured_data(text):
    try:
        # Strip code block formatting
        if text.startswith("```json"):
            text = text[7:].strip("` \n")
        elif text.startswith("```"):
            text = text[3:].strip("` \n")

        # Try to find the first and last curly brace to extract valid JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response.")

        json_str = text[start : end + 1]

        # Optional: Replace smart quotes if they sneak in
        json_str = json_str.replace(""", "\"").replace(""", '"')

        # Optional: Remove trailing commas (invalid in strict JSON)
        import re

        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        data = json.loads(json_str)
        return pd.DataFrame([data])

    except Exception as e:
        print(f"[!] JSON parsing failed: {e}")
        return pd.DataFrame([{"raw_response": text, "error": str(e)}])


# === GEMINI CLIENT SETUP ===
# Model calls will be made through the client

# === SYSTEM PROMPT ===
system_prompt = """ 
Follow instructions precisely as if you were a renowned anthropologist and an economist professor is to determine the following tasks. Organize all the information collected into a JSON object.

For the given paragraph, with temperature = 0, from each paragraph in column "Text” you are tasked to read and interpret each paragraph in the "Text" column using full language comprehension **(without using any outside knowledge, cultural context, or assumptions not directly evidenced by the paragraph itself).** This means: **no reliance on known mythologies, religious systems, or entities (e.g., “Ogoun is a deity”) unless the paragraph itself says so.**

please follow these steps:

1. If the paragraph is not in English, please translate it to English.

2. For the paragraph, identify, if any, all the supernatural beings (including deities, divine or semi-divine beings, ghosts, spirits, souls, monsters, ancestors - only when they play a role when dead - and mediums - priests, nuns, prophets, magicians, spiritists, sorcerers, shamans, lamas, etc.) and magical objects, forces, and places. These are the key elements we are interested in. When you do the identification, limit your answer only to what you can infer from the information of the paragraph, do not use external sources. 

2.1. In a column called ""deities"" create a list of strings of all elements identified in (2), each of them in quotes (ie. ""deity1"") and separated by a comma (ie. ""deity1"", ""deity2"", ""deity3"") in the order in which they appear in the paragraph.

2.2. Add a column called "other names", where you create another list of strings, where the information of each element is separated by a semi-colon, where you add, if any, other names that an element identified **is referred to in the text**, or write "missing" if there are no other names of that element according to the paragraph. If an element has multiple other names, separate them with a comma (ie. missing; "God of the sky", "He who is all powerful"; missing).

3. Classify each element from instruction 2 into ""individual"" if there is only one deity by that name, ""multiple"" if the type of deity identified corresponds to multiple beings, forces or objects, and ""missing"" if you cannot tell from the information from the paragraph. 
3.1. In a column called “cat_type” create a list of strings of all genders identified, in the order of the corresponding deities of 2.1.

4. For each identified element, please estimate if it fits into any of the following categories, using as a general guide the list of suggested keywords, and using your understanding of the paragraph only. Do that for each category below (from 4.1 to 4.20). For each category create a column with a list of strings, where each element of the string represents an element, in the same order as (2.1). Within each column the list of strings should include 1s and 0s. **Only mark categories when the paragraph clearly supports them. If unsure, mark as 0.** Column names should be: cat_creator_universe, cat_creator_human, cat_mother, cat_wife, cat_primal, cat_omni, cat_present, cat_absent, cat_warrior, cat_nature, cat_cosmos, cat_death, cat_ruler, cat_dual, cat_trick, cat_evil, cat_good, cat_demigod, cat_inter, cat_object_force.

4.1. Creator of the universe (“creator”, “founder”, “father”, “mother”, “created”, “conceived”) of/the (“cosmos”, “universe”, “world”, “earth”), 
4.2. Creator of humankind (“creator”, “founder”, “father”, “mother”, “mould”, “created”) of/the (“people”, “race”, “human”, “mankind”, “our mother” or “our father”, “made people”), 
4.3. Mother (of the cosmos, humankind, other gods or specific; “mother”, “first seed”, “cosmic egg”, “womb”,  “her children”, “birthed”, “fertility” or Creator of humankind AND female),
4.4. Wife (Wife of another deity or supernatural or semi-divine being), 
4.5. Primal (“primal”, “oldest”, “prime”, “foundation”), 
4.6. Omnipresent or Omnipotent (“all-wise”, “all-powerful”, “all things”, “all people”),
4.7. Present (“intermediary”, “invocation”, “pay him homage”, “patron of",  “possesses people”, “rituals”),
4.8. Absent (“from afar”, “not intervening”, “distant”, “absence”, “contemplation”, “indifferent”, “watches from”),
4.9. Warrior or hero (“aar”, “hero”, “military”, “battlefield”, “blade”, “blow”, “spear"", “spear”, “blood”, “conquest”, “enemies”, “battle”),
4.10. Related to nature (“fertility”, “seasons”, “water”, “animal”, “river”) 
4.11. Related to cosmos (“sky”, “storm”, “stars”, “sun”, “moon”, “clouds”, “time”), 
4.12. Related to death (“underworld”, “death”, “afterlife”, “judgement”, “souls”, “fate”, “ghost”. It does not include all elements that are dead or kill, but those who are related to death generally) 
4.13. Ruler (“law”, “order”, “sovereign” , “kingship”, “owner”); 
4.14. Dualistic (“opposites”, “twin”, “embrace” , [opposite words in the same sentence], “neither could exist without the other”),
4.15. Trickster ("mischievous", “chaos”, “deceiver”, “deceit”, deities that like to play tricks on humans), 
4.16. Evil (“demon”, “tyranny”, “vengeance”, “devil”, “evil”, “malevolent”, “destructive”), 
4.17. Good (“benevolent”, “mercy”, “nurturing”, “mediator”, “compassion”, “protection, “healing”, “welfare”, “redemption”, “salvation”),
4.18. Demigods (“sacred king”, “ancestor”, “supernatural”, “elf”, “elves”, “ghost”, “embodiment of the god”, “minor divinity”, “giant”, “apparitions”, “spirit”, “nommo”, "soul", “monsters”), 
4.19. Intermediaries (“shaman”, “shamanka”, “priest”, “priestess”, “nun”, “diviner”, “angel”, “saint”, “prophet”, “mystic”, “monk”, “oracle”, “magicians”, “lamas”, “prophets”, beings that are between the supernatural and the human world),
4.20. Sacred places, objects or forces (“shrine”, “totems”, “church”, “juju”, “magic”, or any other object that has a special power or meaning)

5. For each identified element, please infer its or their gender from the paragraph. Do not simply look at the sentence where they appear, but scan the whole paragraph for clues that reveal their gender. With these, create a column “gender”, in the same order as the list in (2.1), list the genders identified as strings (i.e. male, female, general, missing, etc.).
5.1. If cat_type is missing or “individual”, classify its gender into: "male", "female", "androgynous" if it is a single being with both male & female genders, or "genderless" if it does not have any gender.
5.2. If cat_type is “multiple”, classify into: "male" if all members are male, "female" if all members are female.
5.3. If you can't infer the gender or genders from the paragraph, classify it as "missing".  

6. Add two more columns called "certainty_deity" and "certainty_gender" and list, separated by a comma, with the respective certainty levels for each element in the lists of (2.1) and (5 to 5.3), also as a list of strings (ie. "50", "95", "85”). Define each certainty level based on how sure you are of your answer (of the element being an element of interest) and of its gender being correctly identified, using a continuous scale from 0 to 100, where 0 is that you have zero clues and 100 is that you are 100% sure. **Use a lower score if the classification is inferred from suggestive language rather than direct claims.**

7. Create a column called inception_myth where you write 1 if the paragraph is part of a myth of creation, or conception of the world, of humankind, or of a specific group of people, and 0 otherwise. "

8. Create a column called non_english where you write 1 if the paragraph is not in English, and 0 otherwise.
At the end, return only a valid JSON object containing the following fields (with lists for each element if applicable):
```json
{{
  "deities": ["ExampleDeity", "ExampleDeity2"],
  "gender": ["female", "female"],
  "certainty_deity": [95, 95],
  "certainty_gender": [90, 90],
 "other names": ["missing", ["ExampleDeity2_other_name", "ExampleDeity2_other_name_2"]],
  "cat_creator_universe": [1, 1],
  "cat_creator_human": [0, 0],
  "cat_mother": [0, 0],
  "cat_wife": [0, 0],
  "cat_primal": [0, 0],
  "cat_omni": [0, 0],
  "cat_present": [1, 1],
  "cat_absent": [0, 0],
  "cat_warrior": [0, 0],
  "cat_nature": [0, 0],
  "cat_cosmos": [1, 1],
  "cat_death": [0, 0],
  "cat_ruler": [0, 0],
  "cat_dual": [0, 0],
  "cat_trick": [0, 0],
  "cat_evil": [0, 0],
  "cat_good": [1, 1],
  "cat_demigod": [0, 0],
  "cat_inter": [0, 0],
  "cat_object_force": [0, 0],
  "cat_type": ["individual", "individual"],
  "inception_myth": [1, 1],
  "non_english": [0, 0]
}}

"Respond with only the JSON object. Do not include explanations, prefaces, or markdown formatting. Begin your response with a curly brace and end with a closing curly brace."

 
"""

# === MAIN LOOP ===
for i in tqdm(range(last_processed_index + 1, len(df), batch_size), desc="Processing"):
    batch = df.iloc[i : i + batch_size]

    for idx, row in batch.iterrows():
        paragraph = row["Text"]
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\nParagraph: {paragraph}"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8192,  # Adjust as needed
                    top_p=1.0,
                    top_k=1,
                ),
            )

            reply = response.text
            print(f"\n--- Raw Gemini Response ---\n{reply}\n------------------------\n")
            row_data = extract_structured_data(reply)
            row_data["uuid"] = row["uuid"]
            result_df = pd.concat([result_df, row_data], ignore_index=True)

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            # Add a small delay on error to avoid hitting rate limits
            time.sleep(1)
            continue

    # Save progress
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
