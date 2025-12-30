# src/train.py
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from warnings import filterwarnings
filterwarnings('ignore')

# Dynamic import to handle running from root or src/ directory
try:
    from utils import load_data, clean_and_encode_data, save_model_artifact
except ImportError:
    from src.utils import load_data, clean_and_encode_data, save_model_artifact

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_employee_attrition.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_xgb_model.pkl')

def main():
    """
    Main training pipeline using XGBoost:
    1. Loads and cleans data.
    2. Splits into Train/Test sets.
    3. Calculates class imbalance to weight the model.
    4. Optimizes XGBoost hyperparameters using Cross-Validation.
    5. Tunes the decision threshold to maximize F1-Score.
    6. Saves the model artifact for XAI analysis.
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
    
    # Calculate Class Imbalance Ratio for XGBoost
    # Logic: If we have 10 negatives for 1 positive, the positive is 10x more important.
    # Formula: sum(negative instances) / sum(positive instances)
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    imbalance_ratio = neg_count / pos_count
    
    print(f"Class Imbalance Ratio detected: {imbalance_ratio:.2f}")

    # ---------------------------------------------------------
    # 3. Model Training & Hyperparameter Tuning
    # ---------------------------------------------------------
    print("\n--- 3. Training XGBoost with Grid Search & CV ---")
    
    # Initialize XGBoost Classifier
    # use_label_encoder=False avoids a deprecation warning in newer versions
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Hyperparameter Grid
    # We focus on depth and class weight to handle the complex human behavior patterns
    param_grid = {
        'n_estimators': [100, 200],          # Number of boosting rounds
        'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage (lower is more robust)
        'max_depth': [3, 4, 5],              # Shallow trees prevent overfitting
        'subsample': [0.8, 1.0],             # Stochastic sampling
        'scale_pos_weight': [imbalance_ratio, imbalance_ratio * 1.2] # Penalize mistakes on 'Yes'
    }
    
    # 5-Fold Stratified Cross-Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_clf,
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
    # 4. Evaluation & Threshold Tuning
    # ---------------------------------------------------------
    print("\n--- 4. Threshold Optimization & Final Evaluation ---")
    
    # Predict probabilities (needed for threshold tuning)
    y_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate Precision-Recall curve to find the optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 score for each threshold to find the "sweet spot"
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # Handle division by zero
    
    # Find the index of the best F1 score
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1_score = f1_scores[best_idx]
    
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f} -> F1 Score: {best_f1_score:.3f}")
    
    # Generate final predictions using the optimal threshold
    # Note: If prob > threshold then 1 (Leave), else 0 (Stay)
    y_pred_optimized = (y_probs >= optimal_threshold).astype(int)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_optimized))
    
    # ---------------------------------------------------------
    # 5. Save Artifacts
    # ---------------------------------------------------------
    # We save the model AND the feature names (crucial for SHAP/LIME)
    save_model_artifact(best_model, X.columns.tolist(), MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()