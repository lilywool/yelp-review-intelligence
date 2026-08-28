# Yelp Review Intelligence

**Local, reproducible NLP analytics and interactive business intelligence built from the Yelp Open Dataset.**

Yelp Review Intelligence transforms customer reviews into structured NLP features, statistical insights, and interactive business dashboards.

The project focuses on two case-study industries:

* **Mexican restaurants** — used as a proxy for Chipotle
* **Hair salons** — used as a proxy for Great Clips

The source is the **8.6M-review Yelp Open Dataset**. This iteration is designed around reproducible 15,000-review industry samples that can be generated and analyzed directly on a local computer.

---

## Project Evolution

This project has evolved across three iterations during my MSBA program, with each iteration emphasizing a different part of the analytics workflow.

### Iteration 1 — MSBA 502

The first iteration established the analytical foundation using a 15,000-review sample:

* Text preprocessing
* Sentiment analysis
* Emotion analysis
* Feature engineering
* Regression and classification modeling

### Iteration 2 — MSBA 503

The second iteration expanded the project to the **8.6M-review Yelp dataset** and explored distributed data engineering using:

* Apache Spark
* AWS
* Databricks
* Distributed feature engineering
* Large-scale data processing

The focus shifted from analyzing a sample to understanding how an NLP workflow could operate at dataset scale.

### Iteration 3 — Yelp Review Intelligence

This iteration takes the feature-engineering and analytical work developed across the project and rebuilds it around a different objective:

> **Make industry-level Yelp intelligence reproducible, portable, and easy to explore on a local computer.**

Rather than requiring a cloud-based distributed environment, this version provides a self-contained local workflow that:

* Streams the original Yelp JSON dataset
* Generates reproducible industry samples
* Computes review-level NLP features locally
* Incorporates feature-quality checks directly into the processing workflow
* Produces analysis-ready datasets
* Supports statistical modeling
* Powers an interactive Streamlit dashboard
* Enables industry → location → business exploration
* Provides review-level predictive and what-if analysis

The result is a more accessible analytical workflow for exploring **how customer language translates into business insight**.

---

# Project Architecture

Iteration 2 and Iteration 3 serve different purposes.

```text
ITERATION 2
8.6M Yelp Reviews
       │
       ▼
Apache Spark
       │
       ▼
AWS / Databricks
       │
       ▼
Distributed processing
       │
       ▼
Large-scale feature dataset


ITERATION 3
8.6M Yelp Reviews
       │
       ▼
Reproducible local sampling
       │
       ▼
NLP feature engineering
       │
       ▼
Integrated feature-quality checks
       │
       ▼
Analysis-ready datasets
       │
       ├───────────────┐
       ▼               ▼
Statistical        Streamlit
Analysis           Dashboard
       │               │
       └───────┬───────┘
               ▼
       Industry Insights
```

The architectural tradeoff is intentional:

**Iteration 2 prioritizes distributed scale.**

**Iteration 3 prioritizes local reproducibility, analytical iteration, and business usability.**

---

# From Customer Reviews to Business Insight

The workflow connects unstructured customer language to business-facing analysis:

```text
Customer Reviews
       ↓
Text Processing
       ↓
NLP Feature Engineering
       ↓
Sentiment + Emotion + Affect
       ↓
Statistical Analysis
       ↓
Industry / Location / Business Comparisons
       ↓
Interactive Dashboard
       ↓
Business Insight
```

The goal is not simply to classify reviews.

It is to provide a structured way to investigate questions such as:

* How does customer sentiment differ between industries?
* Which emotional patterns characterize positive and negative reviews?
* How does review language relate to star ratings?
* What linguistic characteristics distinguish businesses?
* How do sentiment and affect vary geographically?
* What does a new review's language suggest about its likely rating or sentiment?

---

# Feature Engineering

Each review is transformed into a structured feature vector.

| Category                      | Columns                                                                                                               | Method                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Text**                      | `stripped_review`                                                                                                     | URL stripping, whitespace normalization     |
| **Linguistic**                | `word_count`, `char_count`, `avg_word_len`, `avg_sentence_len`, `sentence_count`                                      | Direct text measurements; spaCy sentencizer |
| **Punctuation / formatting**  | `num_excl`, `num_ques`, `num_caps`, `num_at`, `num_hash`                                                              | Direct counts                               |
| **Negation**                  | `negation_count`                                                                                                      | Negation-term detection                     |
| **Sentiment**                 | `vader_sentiment_score`, `vader_pos`, `vader_neu`, `vader_neg`                                                        | VADER                                       |
| **Emotion**                   | `{emotion}_count`, `{emotion}_int_avg` for 8 emotions, `positive_count`, `negative_count`, `dominant_emotion`         | NRC Emotion Lexicon / NRCLex                |
| **Affect dimensions**         | `Valence_avg`, `Arousal_avg`, `Dominance_avg`, `vad_matched_words`                                                    | NRC-VAD Lexicon v2.1                        |
| **Entities**                  | `person_count`, `location_count`, `product_count`                                                                     | spaCy `en_core_web_sm` NER                  |
| **Transformers** *(optional)* | `hf_sentiment_label`, `hf_sentiment_confidence`, `hf_computed_sentiment`, `hf_emotion_label`, `hf_emotion_confidence` | HuggingFace transformer models              |

---

# NLP Feature Layers

The feature set intentionally combines multiple perspectives on review language.

### Linguistic structure

Captures characteristics such as:

* Review length
* Word length
* Sentence length
* Sentence count
* Punctuation
* Capitalization
* Negation

### Sentiment

VADER provides:

* Positive sentiment
* Neutral sentiment
* Negative sentiment
* Compound sentiment

### Emotion

NRC-based features capture eight emotion categories alongside positive and negative affect.

The pipeline also identifies the dominant emotion for each review.

### Affective dimensions

NRC-VAD expands the analysis beyond categorical emotion with three continuous dimensions:

* **Valence** — pleasantness vs. unpleasantness
* **Arousal** — activation vs. calmness
* **Dominance** — control vs. submission

This provides a richer representation of customer affect than a single positive/negative sentiment score.

### Named entities

spaCy NER captures counts of:

* People
* Locations
* Products

These features provide additional information about what reviewers discuss.

---

# Lexicons and Transformers

The default pipeline uses lightweight NLP methods:

* **VADER** for sentiment
* **NRC Emotion Lexicon / NRCLex** for emotion
* **NRC-VAD** for affective dimensions
* **spaCy** for linguistic processing and NER

These methods are suitable for local execution without requiring large neural-model downloads.

An optional `--transformers` flag adds contextual NLP models:

* DistilBERT sentiment classification
* `j-hartmann/emotion-english-distilroberta-base` emotion classification

The transformer features coexist with the lexicon features, making it possible to compare contextual and lexicon-based representations within the same analytical dataset.

Transformer processing requires additional dependencies:

```bash
pip install transformers torch
```

They are not included in `requirements.txt` by default because PyTorch is a substantial dependency that is unnecessary for the default workflow.

---

# Feature Naming Notes

Some feature names are retained from earlier iterations for schema compatibility.

The `*_int_avg` columns represent the **proportion of emotion-bearing words associated with each emotion**, rather than a word-level intensity average. The original names are retained to preserve compatibility with the earlier feature schema.

Four legacy features whose definitions could not be reliably reconstructed are excluded from the current feature set:

```text
wcst_count
worry_core_count
anxiety_score_avg
yelp_sentiment_avg
```

The current implementation favors explicitly defined and reproducible features.

---

# Repository Structure

```text
pipeline/
├── make_samples.py
├── real_feature_engineering.py
└── pipeline_run_log.json

data/
├── raw/
├── processed/
└── lexicons/

notebooks/
└── 01_analysis.ipynb

dashboard/
├── app.py
└── assets/
    ├── yelp_transparent_logo.png
    ├── yelp_bg_tile.png
    └── make_bg_tile.py

.streamlit/
└── config.toml

docs/
├── Yelp-Review-Intelligence-NLP-Pipeline-v3.pdf
└── add_credit_footer.py

archive/
└── selected artifacts from Iterations 1–2
```

Raw Yelp data, processed datasets, and the NRC-VAD lexicon are excluded from Git because their respective licensing terms do not permit redistribution.

---

# Setup

Create a virtual environment:

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

# Data Setup

## Yelp Open Dataset

Download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and extract the JSON files locally.

For example:

```text
Yelp-JSON/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
├── yelp_academic_dataset_user.json
└── ...
```

The Yelp dataset itself is not included in this repository.

## NRC-VAD Lexicon

Request/download the [NRC-VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) for research use.

Place the file at:

```text
data/lexicons/NRC-VAD-Lexicon-v2.1.txt
```

Expected format:

```text
term    valence    arousal    dominance
```

---

# Step 1: Generate Industry Samples

`pipeline/make_samples.py` streams the Yelp JSON data and generates fixed-seed samples based on business categories.

This allows the full Yelp dataset to serve as the source while keeping the local analytical workflow manageable.

```bash
python pipeline/make_samples.py \
    --json-dir path/to/Yelp-JSON \
    --industry "Mexican:chipotle" "Hair Salons:hair" \
    --n 15000 \
    --seed 222 \
    --outdir data/raw
```

The `--industry` argument uses:

```text
CATEGORY:LABEL
```

`CATEGORY` is substring-matched against Yelp business categories.

`LABEL` determines the output filename.

The example above produces:

```text
chipotle_sample_15000.csv
hair_sample_15000.csv
```

To omit user-level information:

```bash
--no-users
```

### Reproducibility

The sampler uses a fixed random seed.

Therefore:

**identical source data + identical parameters + identical seed = identical samples.**

The Iteration-2 samples remain preserved as historical artifacts. The samples generated by this repository serve as the canonical inputs for the current local workflow.

---

# Step 2: Run Feature Engineering

Run both industry samples with the default feature set:

```powershell
python pipeline/real_feature_engineering.py `
       --files data/raw/chipotle_sample_15k.csv `
                     data/raw/hair_sample_15k.csv `
    --vad-lexicon data/lexicons/NRC-VAD-Lexicon-v2.1.txt `
    --outdir data/processed
```

For a quick 300-row smoke test:

```powershell
python pipeline/real_feature_engineering.py `
       --files data/raw/chipotle_sample_15k.csv `
       --vad-lexicon data/lexicons/NRC-VAD-Lexicon-v2.1.txt `
       --outdir data/processed `
    --sample 300
```

To include transformer features:

```powershell
python pipeline/real_feature_engineering.py `
       --files data/raw/chipotle_sample_15k.csv `
       --vad-lexicon data/lexicons/NRC-VAD-Lexicon-v2.1.txt `
       --outdir data/processed `
    --transformers
```

The pipeline accepts any CSV containing a review-text column.

```text
--text-col
```

defaults to:

```text
raw_review
```

If a `stars` column is present, the pipeline also incorporates the rating signal into feature diagnostics.

Outputs are written as:

```text
{name}_REAL.csv
```

Pipeline execution metadata is recorded in:

```text
pipeline_run_log.json
```

---

# Analysis

The unified analysis notebook is:

```text
notebooks/01_analysis.ipynb
```

It combines:

* Exploratory data analysis
* NLP feature distributions
* Sentiment analysis
* Emotion analysis
* Affect analysis
* Industry comparisons
* Regression
* Binary classification
* Model evaluation
* Feature interpretation

Current results include:

* **Linear regression R² ≈ 0.60** predicting star ratings
* **≈91% accuracy** for positive/negative classification
* Comparative analysis across the two case-study industries

These results describe performance on the project's sampled datasets and are not intended as claims of generalization to all Yelp reviews.

---

# Interactive Dashboard

The Streamlit dashboard translates the analytical dataset into an interactive business-intelligence interface.

The dashboard supports drilldown from:

```text
Industry
   ↓
State
   ↓
City
   ↓
Business
```

### Descriptive

Explore:

* Review volume
* Ratings
* Sentiment
* Emotion
* Linguistic characteristics

### Diagnostic

Investigate:

* Sentiment differences
* Feature relationships
* Industry patterns
* Review-level characteristics

### Predictive

Explore:

* Star prediction
* Positive/negative classification
* Review-level scoring

### Prescriptive

Use the observed patterns to investigate potential areas for business attention.

### What-if review scoring

The dashboard also supports live review scoring, allowing a user to enter review text and inspect its predicted sentiment characteristics.

Run the dashboard with:

```bash
streamlit run dashboard/app.py
```

---

# Local Analytics by Design

The defining characteristic of this iteration is **accessibility**.

The project uses the Yelp dataset at large scale as its source, but the analytical workflow does not require a cloud account, distributed cluster, or specialized infrastructure.

A user can:

1. Obtain the permitted source data.
2. Generate an industry-specific sample.
3. Run the NLP pipeline locally.
4. Produce an analysis-ready feature dataset.
5. Launch the dashboard.
6. Explore businesses and customer sentiment interactively.

This makes the project useful not only as an NLP experiment, but as a **portable framework for industry-level customer review intelligence**.

The same workflow can be adapted to other Yelp business categories by changing the sampling criteria.

---

# Roadmap

* [x] Reproducible sampler from the Yelp JSON dataset
* [x] Local NLP feature-engineering pipeline
* [x] Linguistic feature engineering
* [x] Sentiment analysis
* [x] Emotion analysis
* [x] NRC-VAD affective dimensions
* [x] Named-entity features
* [x] Optional HuggingFace transformer features
* [x] Integrated feature-quality diagnostics
* [x] Pipeline execution logging
* [x] Unified analysis notebook
* [x] Regression and classification modeling
* [x] Industry comparison
* [x] Interactive Streamlit dashboard
* [x] Industry → state → city → business drilldown
* [x] Live review what-if scoring
* [ ] Run transformer features across the complete canonical samples
* [ ] Add lexicon-vs-transformer comparison analysis
* [ ] Expand beyond the current 15k-review samples

---

# Acknowledgments

* **Yelp Inc.** — Yelp Open Dataset
* **Saif M. Mohammad / National Research Council Canada** — NRC Emotion Lexicon and NRC-VAD Lexicon
* **Hutto & Gilbert (2014)** — VADER sentiment analysis
* **spaCy** — NLP processing and named-entity recognition
* **NRCLex** — NRC-based emotion analysis
* **HuggingFace** — optional transformer models
* **Alex Snyder and Eddie Steele** — original project teammates during Iterations 1–2

Iterations 1–2 were collaborative MSBA projects. **Iteration 3 is my own rebuild and extension of the project.**
