# Legacy scripts

Pre-refactor versions of the LLM pipeline, kept for reference only. They used a
single combined identification + classification prompt with free-form JSON
parsing; the current staged pipeline (`deity_analysis_utils.py`) replaced them
with structured outputs. File paths here are placeholders (`path/to/...`) and
the scripts are not maintained.

- `chatgpt_prompt_api.py` — original single-prompt OpenAI pipeline
- `chatgpt_truth_comparison.py` — original ground-truth comparison (now `evaluation_utils.py`)
- `scoring_gpt_v2.py` — intermediate scoring pass
- `gpt_cleaning.py` — raw-output cleanup utility
