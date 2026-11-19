import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from nlp.text_processor import clean_text

def train_model(data_path="data/training_data.csv"):
    df = pd.read_csv(data_path)
    required_columns = {"description", "category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["description", "category"]).copy()
    if len(df) < 2:
        raise ValueError("Training data must contain at least two labeled rows.")

    df['cleaned_text'] = df['description'].apply(clean_text)

    class_counts = df['category'].value_counts()
    stratify_target = (
        df['category']
        if class_counts.min() >= 2 and len(df) >= max(10, class_counts.size * 2)
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['category'],
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    print("Training model...")
    model = pipeline.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = Path("ml_model/classifier.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    train_model()