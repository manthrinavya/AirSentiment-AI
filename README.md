# BrandPulse AI — Twitter US Airline Sentiment Analysis

This project analyzes the **Twitter US Airline Sentiment** dataset (`Tweets.csv`) using:
1. Text preprocessing
2. Classical NLP: TF-IDF + Logistic Regression
3. Deep Learning: Embedding + LSTM
4. Evaluation: Accuracy, F1, classification reports, confusion matrices
5. Streamlit dashboard with simulated live tweets, sentiment distribution, and trend chart

## Dataset
Download `Tweets.csv` from the Twitter US Airline Sentiment dataset and place it here:

`data/raw/Tweets.csv`

Important columns include:
- `text`
- `airline_sentiment`
- `airline_sentiment_confidence`
- `tweet_created`
- `airline`

## Project structure
See the folder tree in the assignment instructions.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## Run the pipeline

From the `BrandPulse_AI` folder:

```bash
python src/preprocessing.py
python src/classical_model.py
python src/lstm_model.py
python src/evaluation.py
```

Then start the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## Output
Generated files will be saved in:
- `data/processed/cleaned_tweets.csv`
- `models/`
- `results/`
- `reports/`

The notebooks are included as documentation/experimentation versions of the same workflow.
