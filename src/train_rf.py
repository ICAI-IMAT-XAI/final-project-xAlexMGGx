import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Dynamic import to handle running from root or src/ directory
try:
    from utils import load_data, clean_and_encode_data, save_model_artifact
except ImportError:
    from src.utils import load_data, clean_and_encode_data, save_model_artifact

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_employee_attrition.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_rf_model.pkl')

def main():
    """
    Main training pipeline using Random Forest:
    1. Loads and cleans data.
    2. Splits into Train/Test sets.
    3. Optimizes Random Forest hyperparameters using Cross-Validation (User Defined Grid).
    4. Evaluates the best model on the Test Set.
    5. Saves the model artifact for XAI analysis.
    """
    
    # ---------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # ---------------------------------------------------------
    print("--- 1. Loading and Preprocessing Data ---")
    df_raw = load_data(DATA_PATH)
    df_processed = clean_and_encode_data(df_raw)
    
    # Separate Features (X) and Target (y)
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']
    
    print(f"Data Shape: {X.shape[0]} rows, {X.shape[1]} features.")

    # ---------------------------------------------------------
    # 2. Train / Test Split
    # ---------------------------------------------------------
    # We use stratification to maintain the class balance in both sets
    print("\n--- 2. Splitting Data (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 3. Model Training & Hyperparameter Tuning
    # ---------------------------------------------------------
    print("\n--- 3. Training Random Forest with Grid Search & CV ---")
    print("(Note: This extensive grid search might take a moment...)")
    
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42)
    
    # User-defined Hyperparameter Grid
    param_grid = {
        'n_estimators': [i*10 for i in range(1, 11)], # Granular search from 1 to 100 trees
        'max_depth': [5, 10, None],      # Control overfitting
        'min_samples_leaf': [5, 10],     # Smoothing leaf nodes
        'class_weight': ['balanced']     # Handle class imbalance
    }
    
    # 5-Fold Stratified Cross-Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1', # Optimizing for F1 Score (Balance of Precision/Recall)
        n_jobs=-1,    # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Internal CV Score (F1): {grid_search.best_score_:.3f}")

    # ---------------------------------------------------------
    # 4. Final Evaluation
    # ---------------------------------------------------------
    print("\n--- 4. Final Evaluation on Test Set ---")
    
    # Standard prediction (default threshold 0.5)
    y_pred = best_model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    # ---------------------------------------------------------
    # 5. Save Artifacts
    # ---------------------------------------------------------
    # We save the model AND the feature names (crucial for SHAP/LIME)
    save_model_artifact(best_model, X.columns.tolist(), MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()