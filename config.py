"""Central path configuration for the eHRAF deity pipeline.

All scripts resolve their input and output locations from this module so the
repository is portable across machines. Large intermediate artifacts (the
concatenated source list and the full text corpus, several hundred MB) are
written to ``data/intermediate/``, which is excluded from version control.

Scripts inside ``code/`` import this module with a two-line bootstrap::

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import ...
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data directories -------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_SAMPLES = DATA_DIR / "samples"
DATA_INTERMEDIATE = DATA_DIR / "intermediate"  # git-ignored; large artifacts

# --- Raw inputs (version-controlled) ----------------------------------------
# Reference list of all cultures in eHRAF World Cultures (from hraf.yale.edu).
CULTURE_SUMMARY_CSV = DATA_RAW / "qrySummary_eHRAF_WorldCultures_Jan2024.csv"
# Manually exported search results for cultures too large for automated export.
MANUAL_CULTURE_EXPORTS = DATA_RAW / "manual_culture_exports"
# Hand-labeled 250-row evaluation sample (metadata + labels only; the eHRAF
# paragraph text itself is licensed and must be regenerated -- see README).
TRUTH_TABLE_CSV = DATA_SAMPLES / "250_sampled_rows.csv"

# --- Canonical intermediate artifacts (produced by code/scraping/) ----------
# Step 01 downloads per-culture CSVs via the browser; move them (plus the
# manual exports above) into this folder before running step 02.
SCRAPED_EXPORTS_DIR = DATA_INTERMEDIATE / "ehraf_exports"
CONCATENATED_SOURCES_CSV = DATA_INTERMEDIATE / "concatenated_sources.csv"  # step 02
MISSING_CULTURES_CSV = DATA_INTERMEDIATE / "missing_cultures.csv"  # step 03
PARAGRAPH_TEXT_CSV = DATA_INTERMEDIATE / "sources_with_text.csv"  # step 04
CLEANED_CORPUS_CSV = DATA_INTERMEDIATE / "corpus_cleaned_subjects.csv"  # step 07
FINAL_CORPUS_CSV = DATA_INTERMEDIATE / "corpus_final_utf8.csv"  # step 08

# --- Outputs -----------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_SUMMARY = OUTPUT_DIR / "figures" / "dataset_summary"
FIGURES_EVAL = OUTPUT_DIR / "figures" / "evaluation"

# Ensure writable directories exist at import time.
for _d in (DATA_INTERMEDIATE, SCRAPED_EXPORTS_DIR, FIGURES_SUMMARY, FIGURES_EVAL):
    _d.mkdir(parents=True, exist_ok=True)
