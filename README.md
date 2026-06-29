# Wild Rift Churn Prediction

This project demonstrates a churn-risk modeling workflow using a synthetic dataset designed to approximate player lifecycle patterns in a fast-paced mobile MOBA environment.

The analysis focuses on early retention risk prediction and compares Random Forest, XGBoost, and LightGBM on a Wild Rift-inspired synthetic behavioral dataset. The project is intended for portfolio demonstration only and does not reproduce internal production logic.

## Project Overview

Player disengagement is a major challenge in live-service games, especially during the early lifecycle. In mobile MOBAs, short match loops, progression pacing, social play, and frustration signals can all influence whether players remain active or begin dropping off.

This repository shows an end-to-end tabular machine learning workflow for churn-risk analysis, including:
- synthetic data generation from seed gaming behavior data
- preprocessing for numeric and categorical features
- model comparison with cross-validation
- final evaluation with classification metrics and confusion matrix
- churn-risk pattern visualization
- SHAP-based feature interpretation

## Repository Structure

```text
wild-rift-churn-prediction/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ synthetic_wr_dataset.csv
├─ notebooks/
│  └─ wr_churn_prediction.ipynb
├─ scripts/
│  └─ generate_synthetic_wr_dataset.py
├─ images/
│  ├─ figure_1_class_balance.png
│  ├─ figure_2_matches_last_7d.png
│  ├─ figure_3_last_login_gap.png
│  ├─ figure_4_region_risk.png
│  ├─ figure_5_model_comparison.png
│  ├─ figure_6_confusion_matrix.png
│  ├─ figure_7_match_volume_risk.png
│  └─ figure_8_feature_importance.png
└─ reports/
   └─ wild-rift-churn-prediction-report.pdf
