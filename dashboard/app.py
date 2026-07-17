"""
Yelp Review Intelligence - Streamlit dashboard (iteration 3).

Realizes iteration 2's yelp_dashboard() drilldown design (industry -> location
-> business) with the four-view structure from the original proposal:
descriptive, diagnostic, predictive, prescriptive - now running on validated
real features instead of the scaffold's `pass` stubs.

Run locally:
    streamlit run dashboard/app.py

Expects processed CSVs in data/processed/ (produced by the pipeline; see README).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Yelp Review Intelligence", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Match any sample size: chipotle_sample_15k_REAL.csv, chipotle_sample_30000_REAL.csv, etc.
INDUSTRY_PATTERNS = {
    "Mexican Restaurants": "chipotle*_REAL.csv",
    "Hair Salons": "hair*_REAL.csv",
}

EMOTIONS = ["joy", "trust", "anticipation", "surprise", "sadness", "fear", "anger", "disgust"]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_industry(industry: str) -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob(INDUSTRY_PATTERNS[industry]))
    if not matches:
        return pd.DataFrame()
    df = pd.read_csv(matches[-1])  # newest-named match if several
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["year_month"] = df["review_date"].dt.to_period("M").astype(str)
    return df


# ---------------------------------------------------------------------------
# Sidebar: the iteration-2 drilldown (industry -> state -> city -> business)
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")
industry = st.sidebar.selectbox("Industry", list(INDUSTRY_PATTERNS.keys()))
df = load_industry(industry)

if df.empty:
    st.error(
        f"No file matching data/processed/{INDUSTRY_PATTERNS[industry]} found. "
        "Run the pipeline first - see README Step 1 & 2."
    )
    st.stop()

states = ["All"] + sorted(df["state"].dropna().unique().tolist())
state = st.sidebar.selectbox("State", states)
if state != "All":
    df = df[df["state"] == state]

cities = ["All"] + sorted(df["city"].dropna().unique().tolist())
city = st.sidebar.selectbox("City", cities)
if city != "All":
    df = df[df["city"] == city]

businesses = ["All"] + sorted(df["name"].dropna().unique().tolist())
business = st.sidebar.selectbox("Business", businesses)
if business != "All":
    df = df[df["name"] == business]

st.sidebar.markdown(f"**{len(df):,} reviews** in current selection")

# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------

st.title("Yelp Review Intelligence")
scope = " → ".join(x for x in [industry, state, city, business] if x != "All")
st.caption(f"Scope: {scope}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews", f"{len(df):,}")
k2.metric("Avg stars", f"{df['stars'].mean():.2f}")
k3.metric("Avg sentiment", f"{df['vader_sentiment_score'].mean():.3f}")
k4.metric("Avg valence", f"{df['Valence_avg'].mean():.3f}")
k5.metric("% 5-star", f"{(df['stars'] == 5).mean():.0%}")

tab_desc, tab_diag, tab_pred, tab_presc = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

# ---------------------------------------------------------------------------
# Descriptive
# ---------------------------------------------------------------------------

with tab_desc:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Star rating distribution")
        st.bar_chart(df["stars"].value_counts().sort_index())
    with c2:
        st.subheader("Dominant emotion")
        st.bar_chart(df["dominant_emotion"].value_counts())

    st.subheader("Average stars over time")
    monthly = df.groupby("year_month").agg(avg_stars=("stars", "mean"), n=("stars", "size"))
    monthly = monthly[monthly["n"] >= 3]  # suppress noisy months
    if len(monthly) > 1:
        st.line_chart(monthly["avg_stars"])
    else:
        st.info("Not enough months in this selection for a trend line.")

# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

with tab_diag:
    st.subheader("What separates negative from positive reviews here?")
    neg = df[df["stars"] <= 2]
    pos = df[df["stars"] >= 4]
    if len(neg) < 5 or len(pos) < 5:
        st.info("Too few reviews in this selection for a reliable contrast (need ≥5 of each).")
    else:
        diag_features = (
            ["vader_sentiment_score", "Valence_avg", "Arousal_avg", "negation_count",
             "word_count", "num_excl", "num_caps"]
            + [f"{e}_int_avg" for e in EMOTIONS]
        )
        contrast = pd.DataFrame({
            "negative (1-2★)": neg[diag_features].mean(),
            "positive (4-5★)": pos[diag_features].mean(),
        })
        contrast["gap"] = contrast["positive (4-5★)"] - contrast["negative (1-2★)"]
        st.dataframe(contrast.round(3).sort_values("gap"))

        st.subheader("Emotion profile: negative vs positive")
        emo_contrast = contrast.loc[[f"{e}_int_avg" for e in EMOTIONS],
                                    ["negative (1-2★)", "positive (4-5★)"]]
        emo_contrast.index = EMOTIONS
        st.bar_chart(emo_contrast)

    with st.expander("Read sample negative reviews (pain-point mining)"):
        for _, row in neg.nlargest(5, "word_count").iterrows():
            st.markdown(f"**{row['stars']:.0f}★ — {row.get('name', '')}** "
                        f"({row.get('city', '')}): {row['stripped_review'][:400]}…")

# ---------------------------------------------------------------------------
# Predictive
# ---------------------------------------------------------------------------

with tab_pred:
    st.subheader("Star-rating model (linear regression on review features)")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X_cols = ["vader_sentiment_score", "Valence_avg", "Arousal_avg", "Dominance_avg",
              "word_count", "avg_word_len", "num_excl", "num_caps", "negation_count"] \
             + [f"{e}_int_avg" for e in EMOTIONS]

    model_df = load_industry(industry)  # fit on full industry, not just the filtered slice
    X = model_df[X_cols].fillna(0)
    y = model_df["stars"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=222)
    model = LinearRegression().fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    st.metric("Model R² (held-out 30%)", f"{r2:.3f}")

    coefs = pd.Series(model.coef_, index=X_cols).sort_values()
    st.bar_chart(coefs)
    st.caption("Coefficients: what pushes a predicted rating up (positive) or down (negative).")

    st.subheader("What-if: predict a rating from a draft review")
    txt = st.text_area("Paste review text to score",
                       placeholder="The staff was friendly but the wait was way too long...")
    if txt.strip():
        try:
            import sys
            sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
            import real_feature_engineering as fe
            if fe.nlp is None:
                lex = Path(__file__).resolve().parent.parent / "data" / "lexicons" / "NRC-VAD-Lexicon-v2_1.txt"
                fe.init_models(vad_lexicon_path=str(lex))
            text = fe.clean_text(txt)
            ling, doc = fe.linguistic_features(text)
            feats = {**ling, **fe.sentiment_features(text), **fe.emotion_features(text),
                     **fe.vad_features(text)}
            x_new = pd.DataFrame([{c: feats.get(c, 0) for c in X_cols}])
            pred = float(np.clip(model.predict(x_new)[0], 1, 5))
            st.success(f"Predicted rating: **{pred:.1f} stars**  "
                       f"(sentiment {feats['vader_sentiment_score']:.2f}, "
                       f"valence {feats['Valence_avg']:.2f})")
        except Exception as e:  # noqa: BLE001 - surface any setup issue to the user
            st.warning(f"Live scoring unavailable ({e}). "
                       "Ensure pipeline dependencies and the VAD lexicon are installed.")

# ---------------------------------------------------------------------------
# Prescriptive
# ---------------------------------------------------------------------------

with tab_presc:
    st.subheader("Where would improvement move the needle most?")
    if business == "All":
        st.info("Select a specific business in the sidebar for targeted recommendations. "
                "Showing selection-level guidance below.")
    target = df

    recs = []
    neg_share = (target["stars"] <= 2).mean()
    if neg_share > 0.25:
        recs.append(f"**High negative share ({neg_share:.0%} of reviews are 1-2★).** "
                    "Review the Diagnostic tab's pain-point samples - recurring themes there "
                    "are the highest-leverage fixes.")
    anger = target["anger_int_avg"].mean()
    baseline_anger = load_industry(industry)["anger_int_avg"].mean()
    if anger > baseline_anger * 1.2:
        recs.append(f"**Anger runs {anger/baseline_anger - 1:.0%} above the industry baseline.** "
                    "Anger language typically flags service/wait issues rather than product quality.")
    negation = target["negation_count"].mean()
    baseline_neg = load_industry(industry)["negation_count"].mean()
    if negation > baseline_neg * 1.2:
        recs.append("**Above-baseline negation density** ('not clean', 'never again') - "
                    "reviewers are contradicting expectations. Check consistency of experience.")
    sent = target["vader_sentiment_score"].mean()
    baseline_sent = load_industry(industry)["vader_sentiment_score"].mean()
    if sent > baseline_sent:
        recs.append(f"**Sentiment ({sent:.2f}) beats the industry baseline ({baseline_sent:.2f}).** "
                    "Amplify what's working: joy-language reviews here often name specific staff "
                    "or menu items - those are marketable strengths.")

    if not recs:
        recs.append("No red flags versus industry baseline in this selection. "
                    "Maintain current operations; monitor the trend line in Descriptive.")
    for r in recs:
        st.markdown(f"- {r}")

    st.caption("Heuristics compare the current selection against its industry baseline on "
               "validated review features. They surface where to look, not verdicts.")
