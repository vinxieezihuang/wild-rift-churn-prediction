# Wild Rift Player Churn Prediction

## Overview
This project evaluates a churn prediction approach for Wild Rift with a focus on Southeast Asian markets, where cold-start behavior, uneven telemetry quality, and regional diversity make prediction more difficult.

The analysis compares three machine learning frameworks:
- Random Forest
- XGBoost
- LightGBM

The final recommendation is LightGBM, based on its balance of recall, training speed, and deployment practicality.

## Problem
Player churn is a major challenge in mobile gaming because many users disengage within their first few sessions. In this context, missing a true churner is often more costly than generating a moderate number of false alarms. The project therefore prioritizes recall, robustness to sparse early-session data, and operational scalability.

## What This Project Covers
- model comparison for churn prediction
- evaluation of recall, AUC, and practical deployment trade-offs
- cold-start and data-quality challenges
- fairness and explainability considerations
- deployment design using SHAP, Looker, and Google Cloud Storage

## Final Recommendation
LightGBM was selected as the preferred model because it offered:
- the strongest recall
- fast training and inference performance
- better suitability for sparse early-session data
- a practical path to explainability through SHAP

## Tools and Skills
- Python
- LightGBM
- XGBoost
- Random Forest
- SHAP
- Machine Learning
- Churn Prediction
- Model Evaluation
- Feature Engineering
- Data Analysis
- Looker
- Google Cloud Storage

## Repository Contents
- `wild-rift-churn-prediction-report.pdf` — full project report
- `README.md` — project summary

## Notes
This is a project-based analysis built around an academic dataset and deployment design assumptions. Business impact estimates are directional and depend on production data quality, retention intervention design, and live deployment conditions.
