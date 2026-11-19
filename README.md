# Expense Categorizer

Streamlit app for categorizing transactions with a text-classification model and storing the reviewed results in SQLite.

## Stack

- Python 3.12
- Streamlit
- pandas
- scikit-learn
- SQLAlchemy
- spaCy
- TextBlob

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python -m nlp.model_train
```

4. Start the app:

```bash
streamlit run app.py
```

## Data

- `data/training_data.csv` is used to train the model.
- `data/transactions.csv` is an example upload file.

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

## Notes

- The app will create the SQLite schema on startup.
- If the spaCy English model is unavailable, feature extraction falls back to a blank English pipeline instead of failing at import time.
- The training pipeline uses TF-IDF plus a balanced logistic regression baseline.

# Expense-Categorizer-NLP-Based-Transaction-Classification-System
