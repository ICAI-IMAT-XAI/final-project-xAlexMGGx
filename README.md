[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/d89f4r04)

# 🧠 Employee Attrition: Prediction & Explainability (XAI)

This project focuses on predicting employee turnover (**Attrition**) using Machine Learning and, crucially, interpreting the model's decisions using **Explainable AI (XAI)** techniques. The goal is to provide HR stakeholders with actionable insights to prevent burnout and retain talent, moving beyond "black box" predictions.

## 📂 Project Overview

* **Goal:** Predict if an employee is likely to leave and explain the *root causes*.
* **Key Challenge:** The dataset is highly imbalanced (~16% attrition), and contains correlated features (Multicollinearity) that complicate interpretation.
* **Approach:** 
    1.  Train a robust **XGBoost** model optimizing for **F1-Score**.
    2.  Apply **SHAP**, **LIME**, and **Permutation Importance** to interpret results.
    3.  Address the **"Age Paradox"** by comparing a *Full Model* (with correlations) vs. a *Reduced Model* (without redundancies).

## 📊 Dataset

* **Source:** IBM HR Analytics Employee Attrition.
* **Size:** 1,470 observations, 35 features.
* **Target:** `Attrition` (Yes/No).
* **Preprocessing:** 
    * Removal of constant columns (`StandardHours`, `Over18`, `EmployeeCount`) and IDs (`EmployeeNumber`).
    * One-Hot Encoding for categorical variables.
    * Analysis of multicollinearity (e.g., `JobLevel` vs. `MonthlyIncome`).

## 🤖 Model & Methodology

Selected **XGBoost** as the core algorithm due to its performance on tabular data and ability to handle class imbalance via `scale_pos_weight`.

* **Evaluation Metric:** **F1-Score (Positive Class)**. Accuracy was discarded due to the class imbalance.
* **Baseline:** Comparing against trivial models (Constant-0 and Constant-1).
* **Feature Selection Strategy:**
    * **Full Model:** Includes all features. Useful for demographic context (e.g., Age interactions).
    * **Reduced Model:** Removes highly correlated features (`Age`, `JobLevel`, `TotalWorkingYears`) to ensure strict feature importance ranking.

## 🔍 XAI Methods Used

1.  **SHAP (Global Interpretation):** To identify macro-level drivers (e.g., `OverTime` as the #1 cause).
2.  **LIME (Local Interpretation):** To analyze specific employee cases (e.g., "High Performer" at risk of burnout).
3.  **Permutation Importance (Sanity Check):** To validate if the model *truly* relies on a feature or if it is just statistical noise. This was key to detecting the multicollinearity issue with `Age`.

## 📁 Repository Structure

### Notebooks
* `0_preprocessing.ipynb`: Data loading, cleaning, EDA and correlation matrix analysis.
* `2_explainability.ipynb`: The core analysis notebook. Contains SHAP plots, LIME instances, and Permutation Importance checks for both the Full and Reduced models.

### Source Code (`src/`)
* `baseline.py`: Script to evaluate dummy classifiers and establish a minimum F1-Score threshold.
* `train_xgb.py`: Training script for the standard XGBoost model.
* `experiment_feature_selection.py`: Experiment script that trains the **Reduced Model** (dropping redundant features) and compares its performance against the standard XGBoost model. It automatically saves the best model if successful.
* `utils.py`: Helper functions for data loading, preprocessing, dropping features, and saving model artifacts.

> **Note:** GenAI was used to enhance code cleanliness and interpretability, ensuring appropriate variable names, comprehensive docstrings, and the removal of redundant logic. It was **not** used for code generation or any other purpose beyond the already mentioned tasks.

## 🚀 Key Findings

* **Burnout is the main driver:** `OverTime` is the strongest predictor of attrition across all methods.
* **Financial Incentives matter:** `MonthlyIncome` and `StockOptionLevel` are effective retention tools.
* **Context vs. Ranking:** Correlated variables (like `Age`) are useful for context but must be removed to get a clean importance ranking in Permutation Importance tests.
