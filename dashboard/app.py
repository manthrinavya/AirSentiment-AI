import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_tweets.csv"
)

TFIDF_FILE = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

CLASSICAL_FILE = os.path.join(
    BASE_DIR, "models", "logistic_regression.pkl"
)

LSTM_FILE = os.path.join(
    BASE_DIR, "models", "lstm_model.keras"
)

TOKENIZER_FILE = os.path.join(
    BASE_DIR, "models", "tokenizer.json"
)

LABEL_FILE = os.path.join(
    BASE_DIR, "models", "label_encoder.json"
)

CONFIG_FILE = os.path.join(
    BASE_DIR, "models", "lstm_config.json"
)

COMPARISON_FILE = os.path.join(
    BASE_DIR, "results", "comparison.csv"
)

st.set_page_config(
    page_title="AirSentiment AI",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_classical():
    vectorizer = joblib.load(TFIDF_FILE)
    model = joblib.load(CLASSICAL_FILE)
    return vectorizer, model

@st.cache_resource
def load_lstm():

    model = tf.keras.models.load_model(LSTM_FILE)

    with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, tokenizer, labels, config

st.title("✈️ AirSentiment AI")

st.subheader("Twitter US Airline Sentiment Analysis")

st.write(
    "Analyze airline-related tweets using NLP, "
    "Machine Learning and Deep Learning."
)

st.divider()

missing = [
    p for p in [
        DATA_FILE,
        TFIDF_FILE,
        CLASSICAL_FILE,
        LSTM_FILE,
        TOKENIZER_FILE,
        LABEL_FILE,
        CONFIG_FILE
    ]
    if not os.path.exists(p)
]

if missing:

    st.error(
        "Required files are missing. "
        "Run the training pipeline first."
    )

    st.code(
        "python src/preprocessing.py\n"
        "python src/classical_model.py\n"
        "python src/lstm_model.py\n"
        "python src/evaluation.py"
    )

    st.stop()

df = load_data()

st.sidebar.header("🎛️ Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Prediction Model",
    [
        "Classical — TF-IDF + Logistic Regression",
        "Deep Learning — LSTM"
    ]
)

airlines = [
    "All"
] + sorted(
    df["airline"].dropna().unique().tolist()
)

airline_filter = st.sidebar.selectbox(
    "Select Airline",
    airlines
)

if airline_filter != "All":

    filtered = df[
        df["airline"] == airline_filter
    ].copy()

else:

    filtered = df.copy()

st.header("🔎 Analyze Your Own Tweet")

st.write(
    "Enter an airline-related tweet below and select a "
    "model from the sidebar to predict its sentiment."
)

tweet = st.text_area(
    "Enter a tweet:",
    placeholder=(
        "Example: The flight was delayed and "
        "customer service was terrible!"
    ),
    height=120
)


if st.button(
    "🔍 Analyze Sentiment",
    type="primary"
):

    if not tweet.strip():

        st.warning(
            "Please enter a tweet to analyze."
        )

    

    elif model_choice.startswith("Classical"):

        vectorizer, model = load_classical()

        X = vectorizer.transform([tweet])

        prediction = model.predict(X)[0]

        probability = float(
            np.max(
                model.predict_proba(X)[0]
            )
        )

        st.success(
            f"Predicted Sentiment: "
            f"**{prediction.upper()}**"
        )

        st.info(
            f"Confidence: "
            f"**{probability:.2%}**"
        )


    else:

        model, tokenizer, labels, config = load_lstm()

        sequences = tokenizer.texts_to_sequences(
            [tweet]
        )

        padded = pad_sequences(
            sequences,
            maxlen=int(config["max_len"]),
            padding="post",
            truncating="post"
        )

        probs = model.predict(
            padded,
            verbose=0
        )[0]

        index = int(
            np.argmax(probs)
        )

        st.success(
            f"Predicted Sentiment: "
            f"**{labels[index].upper()}**"
        )

        st.info(
            f"Confidence: "
            f"**{float(probs[index]):.2%}**"
        )


st.divider()

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Tweets",
    f"{len(filtered):,}"
)

c2.metric(
    "Positive",
    f"{(filtered['airline_sentiment'] == 'positive').sum():,}"
)

c3.metric(
    "Neutral",
    f"{(filtered['airline_sentiment'] == 'neutral').sum():,}"
)

c4.metric(
    "Negative",
    f"{(filtered['airline_sentiment'] == 'negative').sum():,}"
)


st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Sentiment Distribution")

    distribution = (
        filtered["airline_sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    fig = px.pie(
        distribution,
        names="sentiment",
        values="count",
        hole=0.35,
        title="Positive vs Neutral vs Negative"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.subheader("Airline-wise Sentiment")

    if "airline" in filtered.columns:

        airline_sentiment = pd.crosstab(
            filtered["airline"],
            filtered["airline_sentiment"]
        ).reset_index()

        melted = airline_sentiment.melt(
            id_vars="airline",
            var_name="sentiment",
            value_name="count"
        )

        fig2 = px.bar(
            melted,
            x="airline",
            y="count",
            color="sentiment",
            barmode="group",
            title="Sentiment by Airline"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


st.divider()

st.header("📈 Sentiment Trend")

if "tweet_created" in filtered.columns:

    trend_df = filtered.copy()

    trend_df["tweet_created"] = pd.to_datetime(
        trend_df["tweet_created"],
        errors="coerce"
    )

    trend_df = trend_df.dropna(
        subset=["tweet_created"]
    )

    trend_df["date"] = (
        trend_df["tweet_created"].dt.date
    )

    trend = (
        trend_df.groupby(
            ["date", "airline_sentiment"]
        )
        .size()
        .reset_index(name="count")
    )

    fig3 = px.line(
        trend,
        x="date",
        y="count",
        color="airline_sentiment",
        markers=True,
        title="Sentiment Over Time"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

else:

    st.info(
        "Tweet creation date is not available "
        "for trend visualization."
    )


st.divider()

st.header("🔴 Simulated Live Tweet Stream")

st.caption(
    "This simulates incoming tweets using the supplied "
    "Twitter US Airline Sentiment dataset. "
    "It does not connect to X/Twitter."
)


if "stream_running" not in st.session_state:

    st.session_state.stream_running = False


if st.button("▶ Start / Refresh Live Stream"):

    st.session_state.stream_running = True


if st.session_state.stream_running:

    sample_size = min(
        10,
        len(filtered)
    )

    live = filtered.sample(
        sample_size,
        random_state=random.randint(
            1,
            100000
        )
    ).copy()


    if model_choice.startswith("Classical"):

        vectorizer, model = load_classical()

        X = vectorizer.transform(
            live["clean_text"].astype(str)
        )

        live["predicted_sentiment"] = (
            model.predict(X)
        )

    else:

        model, tokenizer, labels, config = load_lstm()

        sequences = tokenizer.texts_to_sequences(
            live["clean_text"].astype(str)
        )

        padded = pad_sequences(
            sequences,
            maxlen=int(config["max_len"]),
            padding="post",
            truncating="post"
        )

        probs = model.predict(
            padded,
            verbose=0
        )

        indexes = np.argmax(
            probs,
            axis=1
        )

        live["predicted_sentiment"] = [
            labels[i]
            for i in indexes
        ]


    display_cols = [

        c for c in [

            "airline",
            "text",
            "airline_sentiment",
            "predicted_sentiment"

        ]

        if c in live.columns
    ]


    st.dataframe(

        live[display_cols].rename(
            columns={

                "airline_sentiment":
                    "Actual Sentiment",

                "predicted_sentiment":
                    "Predicted Sentiment"

            }
        ),

        use_container_width=True,

        hide_index=True

    )


st.divider()

st.header("📈 Model Performance Comparison")

if os.path.exists(COMPARISON_FILE):

    comparison = pd.read_csv(
        COMPARISON_FILE
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Model comparison file is not available. "
        "Run evaluation.py first."
    )


st.divider()

st.caption(
    "✈️ AirSentiment AI | "
    "NLP + TF-IDF + Logistic Regression + "
    "LSTM + Streamlit"
)