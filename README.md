# Yelp Review Intelligence

NLP feature engineering, analysis, and an interactive dashboard built on the
[Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
(8.6M reviews), focused on two case-study industries: **Mexican restaurants
(Chipotle)** and **hair salons (Great Clips)**.

This is the third iteration of a project that began during my MSBA program.
Iteration 1 (MSBA 502) built a working sentiment/emotion analysis and
regression models on a 15k random sample. Iteration 2 (MSBA 503) attempted a
full AWS/Spark pipeline at 8.6M-review scale — and shipped placeholder mock
values in half its feature columns without anyone catching it. Iteration 3
(this repo) fixes that with a validated single-machine pipeline that
**refuses to save output that fails sanity checks**, then rebuilds the
analysis and dashboard on genuinely real features.

## Key lesson baked into this repo

Iteration 2's bug wasn't exotic: a mock UDF (`"STRIPPED TEXT: " + text[:100]`,
hardcoded emotion scores) was written as a development stand-in and never
replaced. The output *looked* plausible — 60 columns, reasonable-looking
numbers — and passed silently into shared team samples. The fix here is
structural, not one-time:

- `validate()` runs after every pipeline execution and checks that
  `word_count` actually tracks real text length, that sentiment correlates
  with star ratings, and that no feature column is frozen at a single value.
- If any check fails, the pipeline **raises and refuses to write the file**.
- Every run is recorded in an append-only `pipeline_run_log.json` with
  timestamps and validation metrics — an audit trail proving the features
  are real.

## Repo structure

```
pipeline/     make_samples.py (Yelp JSON → sample CSVs) +
              real_feature_engineering.py (samples → validated features) + run log
data/         raw/, processed/, lexicons/  (all gitignored — see Data setup)
notebooks/    EDA and modeling notebooks (Jupyter .ipynb)
dashboard/    Streamlit app 
docs/         data dictionary and design notes
archive/      selected artifacts from iterations 1 & 2 (the before/after story)
```

## Features generated (per review)

| Category | Columns | Method |
|---|---|---|
| Text | `stripped_review` (full cleaned text) | URL stripping, whitespace normalization |
| Linguistic | `word_count`, `char_count`, `avg_word_len`, `avg_sentence_len`, `sentence_count`, `num_excl`, `num_ques`, `num_caps`, `num_at`, `num_hash`, `negation_count` | Direct counts; spaCy sentencizer |
| Sentiment | `vader_sentiment_score`, `vader_pos`, `vader_neu`, `vader_neg` | VADER (compound in [-1, 1]) |
| Emotion | `{emotion}_count`, `{emotion}_int_avg` for 8 emotions + `positive_count`, `negative_count`, `dominant_emotion` | NRC Emotion Lexicon (NRCLex) |
| Affect dimensions | `Valence_avg`, `Arousal_avg`, `Dominance_avg`, `vad_matched_words` | NRC-VAD Lexicon v2.1, word-level lookup averaged per review |
| Entities | `person_count`, `location_count`, `product_count` | spaCy `en_core_web_sm` NER |
| Transformers *(optional, `--transformers`)* | `hf_sentiment_label`, `hf_sentiment_confidence`, `hf_computed_sentiment`, `hf_emotion_label`, `hf_emotion_confidence` | HuggingFace DistilBERT sentiment + `j-hartmann/emotion-english-distilroberta-base` |

### Lexicons vs. transformers

The default pipeline uses lexicon methods (VADER, NRC) because they run fast on
any CPU with no model downloads. The `--transformers` flag **additionally** runs
the HuggingFace models from iteration 1 — contextual models that are noticeably
more accurate, especially for emotion (a lexicon can't distinguish "sad they're
closing!" in a glowing 5-star review from genuine sadness). The tradeoff is
compute: roughly 25 minutes per 15k reviews on CPU, fast on GPU. If your
hardware is adequate, `--transformers` is the recommended upgrade; both feature
sets coexist in the output (`hf_*` columns), so you can compare them directly.
`hf_computed_sentiment` uses iteration 1's formula: `(2 × label − 1) × confidence`,
signed to [-1, 1].

Requires `pip install transformers torch` (not in requirements.txt by default,
since torch is a multi-GB install many users won't need).

> **Naming caveat:** `*_int_avg` columns are *proportions* (that emotion's
> share of the review's emotion-bearing words), not word-level intensity
> averages. The name is kept for schema compatibility with iteration 2.
> Four iteration-2 columns with no recoverable definition (`wcst_count`,
> `worry_core_count`, `anxiety_score_avg`, `yelp_sentiment_avg`) were
> deliberately dropped rather than guessed at.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Data setup (not committed to this repo)

Neither the Yelp data nor the NRC lexicon can be redistributed here, so both
are regenerated/downloaded locally:

1. **Yelp data** — download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
   (license restricts redistribution) and unzip the JSON files into a folder,
   e.g. `Yelp-JSON/`.
2. **NRC-VAD Lexicon** — request/download from the
   [official NRC page](https://saifmohammad.com/WebPages/nrc-vad.html)
   (free for research; terms prohibit redistribution). Save as
   `data/lexicons/NRC-VAD-Lexicon-v2_1.txt` (tab-separated:
   `term  valence  arousal  dominance`).

## Step 1: Generate input samples (reproducible)

`pipeline/make_samples.py` streams the raw Yelp JSON (fine on a laptop — two
low-memory passes over the 8.6M reviews) and extracts fixed-seed random samples
per industry. **Identical inputs + identical seed = byte-identical samples**,
which is how this repo stays reproducible without redistributing Yelp's data.

```bash
cd pipeline
python make_samples.py --json-dir path/to/Yelp-JSON \
    --industry "Mexican:chipotle" "Hair Salons:hair" \
    --n 15000 --seed 222 --outdir ../data/raw
```

`--industry` takes `CATEGORY:LABEL` pairs — CATEGORY is substring-matched
against each business's Yelp categories; LABEL names the output file
(`{label}_sample_{n}.csv`). Add `--no-users` to skip the user JSON for a faster
run without user-level columns.

> Historical note: the original iteration-2 samples were drawn by a since-retired
> AWS/Spark pipeline, so they are not bit-reproducible by this script. The
> samples produced here are the canonical iteration-3 inputs.

## Step 2: Run the feature pipeline

```bash
cd pipeline

# Both industry samples, lexicon features (default)
python real_feature_engineering.py \
    --files ../data/raw/chipotle_sample_15000.csv ../data/raw/hair_sample_15000.csv \
    --vad-lexicon ../data/lexicons/NRC-VAD-Lexicon-v2_1.txt \
    --outdir ../data/processed

# Quick smoke test on 300 rows
python real_feature_engineering.py --files ../data/raw/chipotle_sample_15000.csv \
    --vad-lexicon ../data/lexicons/NRC-VAD-Lexicon-v2_1.txt \
    --outdir ../data/processed --sample 300

# With the optional transformer models as well (GPU recommended)
python real_feature_engineering.py --files ../data/raw/chipotle_sample_15000.csv \
    --vad-lexicon ../data/lexicons/NRC-VAD-Lexicon-v2_1.txt \
    --outdir ../data/processed --transformers
```

The pipeline accepts **any CSV with a review-text column** (`--text-col`,
default `raw_review`); a `stars` column additionally enables the
sentiment-vs-rating validation check.

Outputs `{name}_REAL.csv` per input plus an appended entry in
`pipeline_run_log.json`. A failed validation raises an `AssertionError` and
writes nothing.

Typical validation results on the 15k samples:

| Dataset | vader ↔ stars | Valence ↔ stars |
|---|---|---|
| Chipotle (Mexican restaurants) | 0.69 | 0.59 |
| Great Clips (hair salons) | 0.74 | 0.64 |

## Using pipeline functions in notebooks

The module is import-safe (no side effects at import):

```python
from pipeline.real_feature_engineering import init_models, process_dataframe, validate

init_models(vad_lexicon_path="data/lexicons/NRC-VAD-Lexicon-v2_1.txt")
df_features = process_dataframe(df, text_col="raw_review")
validate(df_features)
```

## Roadmap

- [x] Validated feature-engineering pipeline (this repo's core)
- [x] Reproducible sampler from raw Yelp JSON (`make_samples.py`)
- [x] Optional HuggingFace transformer features (`--transformers`)
- [x] Unified analysis notebook (`notebooks/01_analysis.ipynb`) — linear R² = 0.60
      predicting stars, 91% accuracy classifying positive/negative, per-industry comparison
- [x] Streamlit dashboard (`dashboard/app.py`) — industry → state → city → business
      drilldown with descriptive, diagnostic, predictive (incl. live what-if review
      scoring), and prescriptive views. Run: `streamlit run dashboard/app.py`
- [ ] Scale beyond 15k samples (the sampler already streams the full 8.6M;
      just raise `--n`)
- [ ] Lexicon-vs-transformer feature comparison section (needs a `--transformers` run)

## Acknowledgments

- Yelp Open Dataset (Yelp Inc.) — data used under its dataset license
- NRC Emotion Lexicon & NRC-VAD Lexicon (Saif M. Mohammad, National Research
  Council Canada)
- VADER sentiment (Hutto & Gilbert, 2014), spaCy, NRCLex
- Original MSBA project teammates: Alex Snyder and Eddie Steele
  (iterations 1–2 were team efforts; this iteration-3 rebuild is my own)
