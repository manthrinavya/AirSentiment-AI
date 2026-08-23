import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "cleaned_tweets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_WORDS = 20000
MAX_LEN = 60

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Run preprocessing.py first.")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(DATA_FILE)

    X_train, X_test, y_train_text, y_test_text = train_test_split(
        df["clean_text"].astype(str),
        df["airline_sentiment"].astype(str),
        test_size=0.20,
        random_state=42,
        stratify=df["airline_sentiment"]
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test_pad = pad_sequences(
        X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    num_classes = len(label_encoder.classes_)
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LEN),
        LSTM(64),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    probabilities = model.predict(X_test_pad, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print("\n=== Embedding + LSTM ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    ))

    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f)

    with open(os.path.join(MODEL_DIR, "lstm_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "max_words": MAX_WORDS,
            "max_len": MAX_LEN,
            "vocab_size": vocab_size
        }, f, indent=2)

    metrics = pd.DataFrame([{
        "model": "Embedding + LSTM",
        "accuracy": accuracy,
        "weighted_f1": f1
    }])
    metrics.to_csv(
        os.path.join(RESULTS_DIR, "lstm_metrics.csv"),
        index=False
    )

    pd.DataFrame({
        "text": X_test.values,
        "actual": y_test_text.values,
        "predicted": label_encoder.inverse_transform(predictions)
    }).to_csv(
        os.path.join(RESULTS_DIR, "lstm_predictions.csv"),
        index=False
    )

    print("\nLSTM model and tokenizer saved in models/.")

if __name__ == "__main__":
    main()
