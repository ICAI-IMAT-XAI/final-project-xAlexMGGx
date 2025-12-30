import os
import pandas as pd
import joblib


def load_data(filepath):
    """
    Loads the dataset from a specified CSV file path.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")

    print(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)


def clean_and_encode_data(df):
    """
    Performs basic cleaning and preprocessing:
    1. Drops constant or ID columns (noise).
    2. Encodes the target variable 'Attrition' (Yes=1, No=0).
    3. Applies One-Hot Encoding to categorical variables.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame ready for training.
    """
    df_clean = df.copy()

    # 1. Drop columns with no predictive value (constants or IDs)
    # 'EmployeeCount', 'Over18', 'StandardHours' are known constants in this dataset.
    # 'EmployeeNumber' is an ID.
    cols_to_drop = [
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber",
    ]

    # Only drop columns that actually exist to avoid errors
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]

    if existing_cols_to_drop:
        df_clean = df_clean.drop(columns=existing_cols_to_drop)

    # 2. Target Encoding: Attrition (Yes -> 1, No -> 0)
    if "Attrition" in df_clean.columns:
        df_clean["Attrition"] = df_clean["Attrition"].map({"Yes": 1, "No": 0})

    # 3. One-Hot Encoding for categorical features (Department, JobRole, etc.)
    # drop_first=True is used to prevent multicollinearity (dummy variable trap)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    return df_encoded


def drop_features(df, features_list):
    """
    Removes a specific list of features from the DataFrame.
    Used for feature selection experiments (e.g., removing highly correlated variables).

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_list (list): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    # Filter list to only include columns that actually exist in the df
    # This prevents errors if we try to drop a column that was already removed
    existing_cols = [col for col in features_list if col in df.columns]

    if existing_cols:
        print(f"📉 Dropping {len(existing_cols)} features: {existing_cols}")
        return df.drop(columns=existing_cols)
    else:
        print("⚠️ No matching columns found to drop.")
        return df


def save_model_artifact(model, feature_names, filepath):
    """
    Saves the trained model and feature names into a .pkl file.
    Saving feature names is CRITICAL for XAI tools (SHAP/LIME) to work correctly later.

    Args:
        model: Trained sklearn/xgboost model.
        feature_names (list): List of column names used for training.
        filepath (str): Destination path for the .pkl file.
    """
    artifact = {"model": model, "features": feature_names}

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    joblib.dump(artifact, filepath)
    print(f"✅ Model artifact successfully saved to: {filepath}")


def load_model_artifact(filepath):
    """
    Loads the model artifact (dictionary containing model and feature names).

    Args:
        filepath (str): Path to the .pkl file.

    Returns:
        dict: Dictionary with keys 'model' and 'features'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    return joblib.load(filepath)
