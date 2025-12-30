import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dynamic import to handle running from root or src/ directory
try:
    from utils import load_data, clean_and_encode_data
except ImportError:
    from src.utils import load_data, clean_and_encode_data

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_employee_attrition.csv')

def main():
    """
    Baseline Evaluation Script:
    Creates two "dummy" models to establish a performance floor:
    1. All Zeros Model: Predicts everyone stays (Attrition = 0).
       - This highlights the high accuracy caused by class imbalance.
    2. All Ones Model: Predicts everyone leaves (Attrition = 1).
       - This highlights the theoretical minimum precision.
    """
    
    # ---------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # ---------------------------------------------------------
    print("--- 1. Loading Data for Baseline ---")
    try:
        df_raw = load_data(DATA_PATH)
        df_processed = clean_and_encode_data(df_raw)
    except FileNotFoundError as e:
        print(e)
        return

    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']

    # ---------------------------------------------------------
    # 2. Train / Test Split
    # ---------------------------------------------------------
    # Crucial: We use the exact same random_state (42) as train.py
    # to ensure we are comparing against the exact same test set.
    print("\n--- 2. Splitting Data (Same split as train.py) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test Set Size: {len(y_test)}")

    # ---------------------------------------------------------
    # 3. Baseline Model A: Predict All "No" (0)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("BASELINE A: Predict Majority Class (All 'No' / 0)")
    print("="*50)
    
    # Strategy 'constant' with constant=0 forces predictions to always be 0
    dummy_0 = DummyClassifier(strategy='constant', constant=0)
    dummy_0.fit(X_train, y_train)
    y_pred_0 = dummy_0.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_0):.4f}")
    print("Classification Report:")
    # We use zero_division=0 to avoid warnings when precision is 0
    print(classification_report(y_test, y_pred_0, zero_division=0))
    
    print(">> OBSERVATION: Notice high Accuracy but F1-Score of 0.0 for class '1'.")
    print(">> This proves that Accuracy is a bad metric for this imbalanced dataset.")

    # ---------------------------------------------------------
    # 4. Baseline Model B: Predict All "Yes" (1)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("BASELINE B: Predict Minority Class (All 'Yes' / 1)")
    print("="*50)
    
    # Strategy 'constant' with constant=1 forces predictions to always be 1
    dummy_1 = DummyClassifier(strategy='constant', constant=1)
    dummy_1.fit(X_train, y_train)
    y_pred_1 = dummy_1.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_1):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_1, zero_division=0))
    
    print(">> OBSERVATION: Recall is perfect (1.0) because we caught everyone,")
    print(">> but Precision is terrible because we had many false alarms.")

if __name__ == "__main__":
    main()