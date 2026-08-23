import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "cleaned_tweets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Run preprocessing.py first.")

    df = pd.read_csv(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["airline_sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["airline_sentiment"]
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print("\n=== TF-IDF + Logistic Regression ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    metrics = pd.DataFrame([{
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "weighted_f1": f1
    }])
    metrics.to_csv(
        os.path.join(RESULTS_DIR, "classical_metrics.csv"),
        index=False
    )

    pd.DataFrame({
        "text": X_test.values,
        "actual": y_test.values,
        "predicted": predictions
    }).to_csv(
        os.path.join(RESULTS_DIR, "classical_predictions.csv"),
        index=False
    )

    print("\nClassical model saved in models/.")

if __name__ == "__main__":
    main()
