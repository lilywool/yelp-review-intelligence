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
import base64
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Yelp Review Intelligence", layout="wide")

_BG_TILE_PATH = Path(__file__).resolve().parent / "assets" / "yelp_bg_tile.png"
_bg_tile_b64 = base64.b64encode(_BG_TILE_PATH.read_bytes()).decode("ascii")

st.markdown(
    """
    <style>
    :root {
        --ink: #111111;
        --orange: #f06423;
        --orange-dark: #c94914;
        --paper: #f5f1eb;
        --line: #ded7cf;
        --muted: #6f6a64;
    }

    .stApp {
        background-color: var(--paper);
        background-image: url("data:image/png;base64,__BG_TILE_B64__");
        background-repeat: repeat;
        background-size: 340px 340px;
        color: var(--ink);
    }


    [data-testid="stSidebar"] {
        background: var(--ink);
        border-right: 4px solid var(--orange);
    }

    [data-testid="stSidebar"] * {
        color: #f8f5f0;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #242424;
        border-color: #4a4a4a;
    }

    [data-baseweb="popover"] {
        height: 160px !important;
        max-height: 160px !important;
        overflow: hidden !important;
    }

    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] ul {
        height: 160px !important;
        max-height: 160px !important;
        min-height: 0 !important;
        overflow-y: auto !important;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-weight: 800 !important;
        border-left: 10px solid var(--orange) !important;
        padding-left: 28px !important;
        text-transform: uppercase !important;
        margin-top: -8px !important;
        margin-bottom: 24px !important;
        font-size: clamp(0.9rem, 2.4vw, 1.7rem) !important;
        letter-spacing: clamp(0.4px, 0.2vw, 2.4px) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        line-height: 1.3 !important;
        background: rgba(245, 241, 235, 0.88) !important;
        border-radius: 0 4px 4px 0 !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }

    h2, h3 {
        border-left: 6px solid var(--orange) !important;
        padding-left: 20px !important;
        margin-top: 20px !important;
        margin-bottom: 16px !important;
        background: rgba(245, 241, 235, 0.88) !important;
        border-radius: 0 4px 4px 0 !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        display: inline-block !important;
    }

    [data-testid="stSidebar"] h1 {
        margin-bottom: 20px;
        background: none !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 60px !important;
    }

    [data-testid="stCaptionContainer"] {
        margin-top: 4px;
        margin-bottom: 16px;
        background: rgba(255, 255, 255, 0.94) !important;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 2px 6px;
        display: inline-block !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        background-image: none;
        border-top: 4px solid var(--orange);
        border-bottom: 1px solid var(--line);
        border-radius: 4px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.875rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 700;
    }

    [data-baseweb="tab-list"] {
        background: rgba(245, 241, 235, 0.92);
        border-radius: 6px;
        padding: 4px 8px;
    }

    button[data-baseweb="tab"] {
        color: var(--muted);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--orange-dark);
        font-weight: 800;
    }

    [data-baseweb="tab-highlight"] {
        background: var(--orange);
    }

    .stButton > button, .stDownloadButton > button {
        background: var(--orange);
        border-color: var(--orange);
        color: #ffffff;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--orange-dark);
        border-color: var(--orange-dark);
        color: #ffffff;
    }

    [data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }

    [data-testid="stDataFrameResizable"] {
        background: #ffffff !important;
    }

    [data-testid="stAlertContainer"] {
        background: #fdf2ec !important;
        border: 1px solid var(--orange);
        border-radius: 4px;
    }

    [data-testid="stDataFrame"] tbody tr:nth-child(odd) {
        background-color: #faf8f4;
    }

    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #f5f1eb;
    }

    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 4px;
        box-sizing: border-box;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    [data-testid="stExpanderDetails"] {
        background: rgba(255, 255, 255, 0.97);
    }

    .scrim-list {
        background: rgba(245, 241, 235, 0.88);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 14px 18px 14px 34px;
        margin: 8px 0 16px 0;
    }

    .scrim-list li {
        margin-bottom: 6px;
    }
    </style>
    """.replace("__BG_TILE_B64__", _bg_tile_b64),
    unsafe_allow_html=True,
)

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
        rating_counts = df["stars"].value_counts().sort_index().rename_axis("stars").reset_index(name="reviews")
        rating_counts["label"] = rating_counts["stars"].map(lambda value: f"{value:.0f}-star")
        st.bar_chart(rating_counts.set_index("label")["reviews"])
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

    st.subheader("Rating mix and sentiment profile")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        rating_mix = df["stars"].value_counts().sort_index().rename_axis("stars").reset_index(name="reviews")
        rating_mix["label"] = rating_mix["stars"].map(lambda value: f"{value:.0f}-star")
        fig = px.pie(
            rating_mix,
            names="label",
            values="reviews",
            hole=0.58,
            color_discrete_sequence=["#f06423", "#f78c52", "#f5b28d", "#77716b", "#111111"],
        )
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with chart_right:
        sentiment_by_star = (
            df.groupby("stars", as_index=False)
            .agg(avg_sentiment=("vader_sentiment_score", "mean"), reviews=("stars", "size"))
        )
        fig = px.bar(
            sentiment_by_star,
            x="stars",
            y="avg_sentiment",
            text="reviews",
            color="avg_sentiment",
            color_continuous_scale=["#111111", "#f06423"],
            labels={"stars": "Star rating", "avg_sentiment": "Average VADER sentiment"},
        )
        fig.update_traces(texttemplate="n=%{text}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=24, b=36, l=58, r=18),
            height=340,
            yaxis=dict(range=[
                sentiment_by_star["avg_sentiment"].min() - 0.08,
                sentiment_by_star["avg_sentiment"].max() + 0.12,
            ]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
        emo_long = emo_contrast.reset_index(names="emotion").melt(
            id_vars="emotion", var_name="rating group", value_name="avg intensity"
        )
        fig = px.bar(
            emo_long,
            x="emotion",
            y="avg intensity",
            color="rating group",
            barmode="group",
            color_discrete_map={"negative (1-2★)": "#111111", "positive (4-5★)": "#f06423"},
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=48, r=18),
            height=340,
            yaxis=dict(rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Language signals by rating")
        rating_features = (
            df.groupby("stars", as_index=False)
            .agg(sentiment=("vader_sentiment_score", "mean"),
                 valence=("Valence_avg", "mean"),
                 review_length=("word_count", "mean"),
                 reviews=("stars", "size"))
        )
        fig = px.scatter(
            rating_features,
            x="sentiment",
            y="valence",
            size="reviews",
            color="stars",
            text="stars",
            color_continuous_scale=["#111111", "#f06423"],
            labels={"sentiment": "Average VADER sentiment", "valence": "Average valence", "stars": "Stars"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
                lex = Path(__file__).resolve().parent.parent / "data" / "lexicons" / "NRC-VAD-Lexicon-v2.1.txt"
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
    recs_html = "".join(
        f"<li>{re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', r)}</li>" for r in recs
    )
    st.markdown(
        f'<ul class="scrim-list">{recs_html}</ul>',
        unsafe_allow_html=True,
    )

    st.subheader("Priority matrix")
    baseline = load_industry(industry)
    priority = pd.DataFrame([
        {"Signal": "Negative review share", "Current": neg_share,
         "Baseline": (baseline["stars"] <= 2).mean(), "Direction": "Lower is better"},
        {"Signal": "Anger language", "Current": anger,
         "Baseline": baseline["anger_int_avg"].mean(), "Direction": "Lower is better"},
        {"Signal": "Negation density", "Current": negation,
         "Baseline": baseline["negation_count"].mean(), "Direction": "Lower is better"},
        {"Signal": "VADER sentiment", "Current": sent,
         "Baseline": baseline["vader_sentiment_score"].mean(), "Direction": "Higher is better"},
    ])
    priority["Gap vs baseline"] = priority["Current"] - priority["Baseline"]
    priority["Priority"] = np.where(
        ((priority["Direction"] == "Lower is better") & (priority["Gap vs baseline"] > 0)) |
        ((priority["Direction"] == "Higher is better") & (priority["Gap vs baseline"] < 0)),
        "Review",
        "Monitor",
    )
    st.dataframe(
        priority.style.format({"Current": "{:.3f}", "Baseline": "{:.3f}", "Gap vs baseline": "{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Location opportunity map")
    city_summary = (
        target.dropna(subset=["city"])
        .groupby("city", as_index=False)
        .agg(
            reviews=("stars", "size"),
            negative_share=("stars", lambda values: (values <= 2).mean()),
            avg_stars=("stars", "mean"),
            avg_sentiment=("vader_sentiment_score", "mean"),
        )
    )
    city_summary = city_summary[city_summary["reviews"] >= 20]
    if len(city_summary) < 2:
        st.info("Select a broader location scope to compare multiple cities.")
    else:
        fig = px.scatter(
            city_summary,
            x="avg_stars",
            y="negative_share",
            size="reviews",
            color="avg_sentiment",
            hover_name="city",
            hover_data={"reviews": True, "avg_stars": ":.2f", "negative_share": ":.1%"},
            color_continuous_scale=["#111111", "#f06423"],
            labels={
                "avg_stars": "Average stars",
                "negative_share": "Negative review share",
                "avg_sentiment": "Average sentiment",
                "reviews": "Reviews",
            },
        )
        fig.update_layout(
            coloraxis_colorbar_title="Sentiment",
            margin=dict(t=16, b=44, l=58, r=18),
            height=380,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("Heuristics compare the current selection against its industry baseline on "
               "validated review features. They surface where to look, not verdicts.")
