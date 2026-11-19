import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path
from database.db import SessionLocal, init_db
from database.models import Transaction
from nlp.text_processor import clean_text

if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame()

MODEL_PATH = Path("ml_model/classifier.pkl")
EXPECTED_CATEGORY_OPTIONS = ["Food", "Transport", "Entertainment", "Shopping", "Utilities", "Other"]

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None

def regex_categorizer(description):
    patterns = {
        "Food": [r"mcdonald|starbucks|kfc|burger|restaurant"],
        "Transport": [r"uber|lyft|taxi|shell|gas"],
        "Entertainment": [r"netflix|spotify|movie|cinema"],
        "Shopping": [r"amazon|ebay|walmart|target"]
    }
    cleaned = clean_text(description)
    for category, regex_list in patterns.items():
        if any(re.search(pattern, cleaned) for pattern in regex_list):
            return category
    return "Other"

def predict_category(description, model):
    if pd.isna(description) or not str(description).strip():
        return "Uncategorized"

    regex_pred = regex_categorizer(description)
    if regex_pred != "Other":
        return regex_pred
    if model:
        return model.predict([clean_text(description)])[0]
    return "Uncategorized"

def main():
    st.title("💰 Smart Expense Categorizer")
    st.markdown("Upload your transactions and get automatic categorization using NLP and ML")

    init_db()
    model = load_model()
    db = SessionLocal()

    try:
        uploaded_file = st.file_uploader("Upload transactions (CSV)", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Unable to read CSV file: {exc}")
                return

            if df.empty:
                st.warning("The uploaded CSV has no rows.")
                return

            st.session_state.transactions = df

            description_column = next((column for column in df.columns if column.lower() == 'description'), None)
            amount_column = next((column for column in df.columns if column.lower() == 'amount'), None)

            if not description_column:
                st.error("The uploaded CSV must contain a Description column.")
                return

            if amount_column is None:
                df['Amount'] = 0.0
                amount_column = 'Amount'

            with st.spinner("Categorizing transactions..."):
                df['Predicted Category'] = df[description_column].apply(
                    lambda value: predict_category(value, model)
                )

            edited_df = st.data_editor(
                df,
                column_config={
                    "Predicted Category": st.column_config.SelectboxColumn(
                        "Confirm/Correct Category",
                        options=EXPECTED_CATEGORY_OPTIONS
                    )
                },
                num_rows="fixed",
                hide_index=True
            )

            if st.button("Save to Database"):
                saved_count = 0
                for _, row in edited_df.iterrows():
                    description = row.get(description_column)
                    if pd.isna(description) or not str(description).strip():
                        continue

                    raw_amount = row.get(amount_column, 0)
                    amount = float(raw_amount) if pd.notna(raw_amount) and str(raw_amount).strip() else 0.0
                    transaction = Transaction(
                        description=str(description).strip(),
                        amount=amount,
                        predicted_category=row['Predicted Category'],
                        category=row['Predicted Category']
                    )
                    db.add(transaction)
                    saved_count += 1
                db.commit()
                st.success(f"{saved_count} transactions saved to database!")

        if st.checkbox("Show saved transactions"):
            transactions = db.query(Transaction).order_by(Transaction.created_at.desc()).all()
            if transactions:
                display_df = pd.DataFrame([{
                    "Description": t.description,
                    "Amount": t.amount,
                    "Category": t.category,
                    "Date": t.created_at
                } for t in transactions])
                st.dataframe(display_df)

                csv = display_df.to_csv(index=False)
                st.download_button(
                    "Export as CSV",
                    data=csv,
                    file_name="categorized_expenses.csv",
                    mime="text/csv"
                )
            else:
                st.info("No transactions in database yet")
    finally:
        db.close()

if __name__ == "__main__":
    main()