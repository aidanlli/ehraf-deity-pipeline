"""
Deity Analysis Utilities

This module provides utility functions for the deity identification and classification pipeline.
It includes functions for language detection, translation, data parsing, and API interactions
with OpenAI's structured output features.

Functions:
    - detect_language: Detect the language of input text
    - translate_text: Translate text to English using deep-translator
    - calculate_english_percentage: Calculate percentage of English words in text
    - should_translate: Determine if text needs translation based on threshold
    - extract_structured_data: Parse JSON responses from API
    - create_identification_prompt: Create prompt for deity identification
    - create_classification_prompt: Create prompt for deity classification
    - call_openai_api: Make API calls with structured output format

Author: Aidan Li
Date: January 2026
"""

from typing import Any

import openai
import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect
from pydantic import BaseModel
from tqdm import tqdm


class DeityIdentification(BaseModel):
    """Pydantic model for deity identification structured output."""

    deities: list[str]
    other_names: list[str]
    certainty_deity: list[int]
    non_english: int


class DeityClassification(BaseModel):
    """Pydantic model for deity classification structured output."""

    gender: list[str]
    certainty_gender: list[int]
    cat_creator_universe: list[int]
    cat_creator_human: list[int]
    cat_mother: list[int]
    cat_wife: list[int]
    cat_primal: list[int]
    cat_omni: list[int]
    cat_present: list[int]
    cat_absent: list[int]
    cat_warrior: list[int]
    cat_nature: list[int]
    cat_cosmos: list[int]
    cat_death: list[int]
    cat_ruler: list[int]
    cat_dual: list[int]
    cat_trick: list[int]
    cat_evil: list[int]
    cat_good: list[int]
    cat_demigod: list[int]
    cat_inter: list[int]
    cat_object_force: list[int]
    cat_type: list[str]
    inception_myth: int


def detect_language(text: str) -> tuple[str, bool]:
    """
    Detect the language of the input text.

    Args:
        text (str): Input text to analyze

    Returns:
        Tuple[str, bool]: Language code (e.g., 'en', 'es') and success flag
    """
    try:
        lang = detect(text)
        return lang, True
    except LangDetectException:
        return "unknown", False


def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translate text to English using Google Translator via deep-translator.

    Args:
        text (str): Text to translate
        source_lang (str): Source language code (default: 'auto')
        target_lang (str): Target language code (default: 'en')

    Returns:
        str: Translated text
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails


def calculate_english_percentage(text: str) -> float:
    """
    Calculate the percentage of English words in the text.

    This is a simple heuristic that checks if words exist in a basic English word list
    or uses character-based detection.

    Args:
        text (str): Input text to analyze

    Returns:
        float: Percentage of text that appears to be English (0-100)
    """
    # Simple heuristic: detect if text is primarily ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text)

    if total_chars == 0:
        return 0.0

    return (ascii_chars / total_chars) * 100


def should_translate(text: str, threshold: float = 70.0) -> tuple[bool, str, float]:
    """
    Determine if text should be translated based on English content threshold.

    Args:
        text (str): Input text to analyze
        threshold (float): Minimum percentage of English content (default: 70.0)

    Returns:
        Tuple[bool, str, float]: (needs_translation, detected_language, english_percentage)
    """
    # Handle missing or non-string values
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return False, "unknown", 100.0

    lang, success = detect_language(text)
    english_pct = calculate_english_percentage(text)

    if not success:
        return False, "unknown", english_pct

    # If detected as English or high English percentage, don't translate
    if lang == "en" or english_pct >= threshold:
        return False, lang, english_pct

    return True, lang, english_pct


def create_identification_prompt(text: str) -> str:
    """
    Create the system and user prompts for deity identification stage.

    Args:
        text (str): The paragraph text to analyze

    Returns:
        str: System prompt for deity identification
    """
    system_prompt = """
You are a renowned anthropologist tasked with identifying supernatural beings from text passages.

Your task is to identify all supernatural beings (including deities, divine or semi-divine beings, 
ghosts, spirits, souls, monsters, ancestors - only when they play a role when dead - and mediums - 
priests, nuns, prophets, magicians, spiritists, sorcerers, shamans, lamas, etc.) and magical objects, 
forces, and places.

**CRITICAL**: Use ONLY information directly stated in the paragraph. Do not use external knowledge, 
cultural context, or assumptions not evidenced by the text itself.

For each element you identify:
1. List the name/term as it appears in the text
2. Note any other names the element is referred to in the text (or "missing" if none)
3. Rate your certainty (0-100) based on how clear the text is

Return your response as a structured JSON object with these fields:
- deities: List of identified supernatural beings/objects (as strings)
- other_names: List of alternative names mentioned in text (use "missing" if none)
- certainty_deity: List of certainty scores (0-100) for each identification
- non_english: 1 if the original text was not in English, 0 otherwise
"""
    return system_prompt


def create_classification_prompt(text: str, deities: list[str]) -> str:
    """
    Create the system and user prompts for deity classification stage.

    Args:
        text (str): The paragraph text to analyze
        deities (List[str]): List of identified deities from stage 2

    Returns:
        str: System prompt for deity classification
    """
    deity_list = ", ".join([f'"{d}"' for d in deities])

    system_prompt = f"""
You are a renowned anthropologist and economist professor tasked with classifying supernatural beings.

You have been provided with the following deities/supernatural elements identified from a text:
{deity_list}

Based ONLY on the information in the provided text, classify each deity according to the following 
categories. **Only mark categories when the text clearly supports them. If unsure, mark as 0.**

For each deity, determine:

1. GENDER: Infer gender from the text (male, female, androgynous, genderless, missing)
2. CERTAINTY_GENDER: Your certainty about the gender classification (0-100)

3. CATEGORIES (1 if applicable, 0 if not):
   - cat_creator_universe: Creator of the universe/cosmos/world
   - cat_creator_human: Creator of humankind/people
   - cat_mother: Mother figure (of cosmos, humankind, gods, or specific)
   - cat_wife: Wife of another deity/supernatural being
   - cat_primal: Primal/oldest/prime/foundation deity
   - cat_omni: Omnipresent or Omnipotent
   - cat_present: Present/active (rituals, invocations, interventions)
   - cat_absent: Absent/distant/not intervening
   - cat_warrior: Warrior or hero
   - cat_nature: Related to nature (fertility, seasons, water, animals)
   - cat_cosmos: Related to cosmos (sky, storm, stars, sun, moon)
   - cat_death: Related to death/underworld/afterlife
   - cat_ruler: Ruler (law, order, sovereign, kingship)
   - cat_dual: Dualistic (opposites, twins)
   - cat_trick: Trickster (mischievous, chaos, deceiver)
   - cat_evil: Evil (demon, tyranny, malevolent)
   - cat_good: Good (benevolent, mercy, nurturing, healing)
   - cat_demigod: Demigods (ancestors, spirits, minor divinity, giants)
   - cat_inter: Intermediaries (shaman, priest, angel, prophet)
   - cat_object_force: Sacred places, objects, or forces

4. CAT_TYPE: "individual" (single being), "multiple" (multiple beings), or "missing"

5. INCEPTION_MYTH: 1 if the text describes a creation/inception myth, 0 otherwise

Return a structured JSON object with all fields as lists (in same order as deities provided).
"""
    return system_prompt


def call_openai_api(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    response_format: type[BaseModel],
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0,
) -> dict[str, Any]:
    """
    Call OpenAI API with structured output format.

    Args:
        client: OpenAI client instance
        system_prompt (str): System message content
        user_prompt (str): User message content
        response_format: Pydantic BaseModel class for structured output
        model (str): OpenAI model to use
        temperature (float): Sampling temperature

    Returns:
        Dict[str, Any]: Parsed response as dictionary
    """
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            temperature=temperature,
        )

        return response.choices[0].message.parsed.dict()

    except Exception as e:
        print(f"API call error: {e}")
        return {}


def process_translation_stage(
    df: pd.DataFrame, text_column: str = "Text", threshold: float = 70.0
) -> pd.DataFrame:
    """
    Process the translation stage for all rows in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with text column
        text_column (str): Name of the column containing text to analyze
        threshold (float): English percentage threshold for translation

    Returns:
        pd.DataFrame: DataFrame with translation info and translated text
    """
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Translation Stage"):
        text = row[text_column]
        needs_trans, lang, eng_pct = should_translate(text, threshold)

        translated_text = text
        if needs_trans:
            translated_text = translate_text(text, source_lang=lang)

        results.append(
            {
                "uuid": row["uuid"],
                "original_text": text,
                "detected_language": lang,
                "english_percentage": eng_pct,
                "was_translated": needs_trans,
                "translated_text": translated_text,
            }
        )

    return pd.DataFrame(results)


def process_identification_stage(
    df: pd.DataFrame,
    client: openai.OpenAI,
    text_column: str = "translated_text",
    model: str = "gpt-4o-2024-08-06",
) -> pd.DataFrame:
    """
    Process the deity identification stage for all rows.

    Args:
        df (pd.DataFrame): Input dataframe with translated text
        client: OpenAI client instance
        text_column (str): Name of column containing text to analyze
        model (str): OpenAI model to use

    Returns:
        pd.DataFrame: DataFrame with identified deities
    """
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Identification Stage"):
        text = row[text_column]
        system_prompt = create_identification_prompt(text)
        user_prompt = f"Paragraph: {text}"

        response = call_openai_api(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=DeityIdentification,
            model=model,
        )

        result = {
            "uuid": row["uuid"],
            "deities": response.get("deities", []),
            "other_names": response.get("other_names", []),
            "certainty_deity": response.get("certainty_deity", []),
            "non_english": response.get("non_english", 0),
        }
        results.append(result)

    return pd.DataFrame(results)


def process_classification_stage(
    df: pd.DataFrame,
    translation_df: pd.DataFrame,
    identification_df: pd.DataFrame,
    client: openai.OpenAI,
    model: str = "gpt-4o-2024-08-06",
) -> pd.DataFrame:
    """
    Process the deity classification stage for all rows.

    Args:
        df (pd.DataFrame): Original dataframe
        translation_df (pd.DataFrame): Results from translation stage
        identification_df (pd.DataFrame): Results from identification stage
        client: OpenAI client instance
        model (str): OpenAI model to use

    Returns:
        pd.DataFrame: Complete dataframe with all stages combined
    """
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classification Stage"):
        uuid = row["uuid"]

        # Get translated text and identified deities
        trans_row = translation_df[translation_df["uuid"] == uuid].iloc[0]
        ident_row = identification_df[identification_df["uuid"] == uuid].iloc[0]

        text = trans_row["translated_text"]
        deities = ident_row["deities"]

        # Skip if no deities identified
        if not deities or len(deities) == 0:
            # Create empty classification result
            result = {
                "uuid": uuid,
                "gender": [],
                "certainty_gender": [],
                "cat_creator_universe": [],
                "cat_creator_human": [],
                "cat_mother": [],
                "cat_wife": [],
                "cat_primal": [],
                "cat_omni": [],
                "cat_present": [],
                "cat_absent": [],
                "cat_warrior": [],
                "cat_nature": [],
                "cat_cosmos": [],
                "cat_death": [],
                "cat_ruler": [],
                "cat_dual": [],
                "cat_trick": [],
                "cat_evil": [],
                "cat_good": [],
                "cat_demigod": [],
                "cat_inter": [],
                "cat_object_force": [],
                "cat_type": [],
                "inception_myth": 0,
            }
        else:
            system_prompt = create_classification_prompt(text, deities)
            user_prompt = f"Text: {text}\n\nDeities to classify: {', '.join(deities)}"

            response = call_openai_api(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=DeityClassification,
                model=model,
            )

            result = {"uuid": uuid}
            result.update(response)

        results.append(result)

    return pd.DataFrame(results)
