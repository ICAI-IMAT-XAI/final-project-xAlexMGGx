# src/experiment_feature_selection.py
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Dynamic import to handle running from root or src/ directory
try:
    from utils import load_data, clean_and_encode_data, save_model_artifact, drop_features
except ImportError:
    from src.utils import load_data, clean_and_encode_data, save_model_artifact, drop_features

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_employee_attrition.csv')
MODEL_PATH_REDUCED = os.path.join(BASE_DIR, 'models', 'best_xgb_reduced.pkl')

# --- HIGH VIF FEATURES CANDIDATES ---
# These are the variables identified as highly correlated/redundant in the Preprocessing step.
# Removing them reduces noise for SHAP/LIME without necessarily hurting prediction power.
FEATURES_TO_DROP = [
    'JobLevel',           # Redundant with MonthlyIncome
    'TotalWorkingYears',  # Redundant with JobLevel
    'Age'
]

def train_and_get_model(X, y):
    """
    Trains an XGBoost model on the provided dataset and returns performance metrics.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    Returns:
        tuple: (trained_model, f1_score, report_str, feature_names_list)
    """
    # 1. Split (80% Train, 20% Test)
    # Stratify is crucial here to keep the imbalance ratio consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Calculate Class Imbalance Ratio for XGBoost
    # Formula: Negatives / Positives
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    imbalance_ratio = neg_count / pos_count
    
    # 3. Train XGBoost (Standard Hyperparameters for this dataset)
    clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=imbalance_ratio, # Handling imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, f1, report, X_train.columns.tolist()

def main():
    print("--- 🧪 Multicollinearity Reduction & Saving Experiment ---\n")
    
    # 1. Load & Clean Data (Basic Preprocessing)
    df_raw = load_data(DATA_PATH)
    df_processed = clean_and_encode_data(df_raw)
    
    # Separate Target
    X_full = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']
    
    # ---------------------------------------------------------
    # SCENARIO A: BASELINE (All Features)
    # ---------------------------------------------------------
    print(f"👉 Scenario A: Baseline ({X_full.shape[1]} features)...")
    _, f1_a, _, _ = train_and_get_model(X_full, y)
    print(f"   > Baseline F1-Score: {f1_a:.4f}")

    # ---------------------------------------------------------
    # SCENARIO B: REDUCED (Removing High VIF)
    # ---------------------------------------------------------
    print(f"\n👉 Scenario B: Reduced (Attempting to drop {FEATURES_TO_DROP})...")
    
    # Use our custom function from utils to drop features safely
    X_reduced = drop_features(X_full, FEATURES_TO_DROP)
    
    print(f"   > New Feature Count: {X_reduced.shape[1]}")
    
    model_b, f1_b, report_b, features_b = train_and_get_model(X_reduced, y)
    print(f"   > Reduced F1-Score:  {f1_b:.4f}")

    # ---------------------------------------------------------
    # FINAL COMPARISON & SAVING LOGIC
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("📊 EXPERIMENT RESULTS")
    print("="*60)
    
    diff = f1_b - f1_a
    print(f"Performance Change (F1): {diff:+.4f}")
    
    # Decision Rule: 
    # We accept the reduced model if F1 improves OR if it drops very slightly (e.g. > -0.02).
    # Explainability gains (removing multicollinearity) are worth a tiny drop in accuracy.
    if diff >= -0.02:
        print("\n✅ VERDICT: SUCCESS. The Reduced Model is robust.")
        print("   The model maintains performance with fewer redundant variables.")
        print(f"💾 Saving Reduced Model to: {MODEL_PATH_REDUCED}")
        
        save_model_artifact(model_b, features_b, MODEL_PATH_REDUCED)
        
        print("\n[NEXT STEP] Update your '2_explainability.ipynb' to load:")
        print(f"           model_path = '../models/best_xgb_reduced.pkl'")
    else:
        print("\n❌ VERDICT: CAUTION. Significant performance loss detected.")
        print("   The dropped features contained unique information vital for the model.")
        print("   We recommend sticking to the original 'best_xgb_model.pkl'.")
    
    print("\n--- Detailed Report (Reduced Model) ---")
    print(report_b)

if __name__ == "__main__":
    main()