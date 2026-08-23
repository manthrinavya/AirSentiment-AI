import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "Tweets.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_tweets.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    """Clean a tweet while preserving useful sentiment words."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def main():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Download Tweets.csv from the Twitter US Airline Sentiment dataset "
            "and place it in data/raw/."
        )

    df = pd.read_csv(RAW_FILE)

    required = {"text", "airline_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["text", "airline_sentiment"] + [
        c for c in ["airline", "tweet_created", "airline_sentiment_confidence"]
        if c in df.columns
    ]].copy()

    df = df.dropna(subset=["text", "airline_sentiment"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Preprocessing completed.")
    print(f"Rows saved: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nSentiment distribution:")
    print(df["airline_sentiment"].value_counts())

if __name__ == "__main__":
    main()
