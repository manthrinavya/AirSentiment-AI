import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "cleaned_tweets.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

def save_confusion_matrix(prediction_file, output_file, title):
    path = os.path.join(RESULTS_DIR, prediction_file)
    if not os.path.exists(path):
        print(f"Skipping {title}: {prediction_file} not found.")
        return

    df = pd.read_csv(path)
    labels = sorted(set(df["actual"].astype(str)) | set(df["predicted"].astype(str)))
    cm = confusion_matrix(df["actual"], df["predicted"], labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, output_file), dpi=200)
    plt.close()

def main():
    metric_files = []
    for filename in ["classical_metrics.csv", "lstm_metrics.csv"]:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            metric_files.append(pd.read_csv(path))

    if metric_files:
        comparison = pd.concat(metric_files, ignore_index=True)
        comparison.to_csv(
            os.path.join(RESULTS_DIR, "comparison.csv"),
            index=False
        )
        print("\n=== Model Comparison ===")
        print(comparison.to_string(index=False))

    save_confusion_matrix(
        "classical_predictions.csv",
        "classical_confusion_matrix.png",
        "Classical Model Confusion Matrix"
    )
    save_confusion_matrix(
        "lstm_predictions.csv",
        "lstm_confusion_matrix.png",
        "LSTM Confusion Matrix"
    )

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)

        # Overall sentiment distribution
        counts = df["airline_sentiment"].value_counts()

        plt.figure(figsize=(7, 5))
        counts.plot(kind="bar")
        plt.title("Twitter US Airline Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Tweets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, "sentiment_distribution.png"),
            dpi=200
        )
        plt.close()

        # If tweet_created exists, create a time-series trend.
        if "tweet_created" in df.columns:
            dates = pd.to_datetime(df["tweet_created"], errors="coerce")
            temp = df.copy()
            temp["date"] = dates
            temp = temp.dropna(subset=["date"])
            trend = temp.groupby(
                [temp["date"].dt.date, "airline_sentiment"]
            ).size().unstack(fill_value=0)

            plt.figure(figsize=(10, 5))
            trend.plot(ax=plt.gca())
            plt.title("Sentiment Trend Over Time")
            plt.xlabel("Date")
            plt.ylabel("Tweet Count")
            plt.tight_layout()
            plt.savefig(
                os.path.join(RESULTS_DIR, "sentiment_trend.png"),
                dpi=200
            )
            plt.close()

    print("\nEvaluation completed. Check the results/ folder.")

if __name__ == "__main__":
    main()
