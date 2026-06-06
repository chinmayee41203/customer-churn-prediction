# Customer Churn Prediction

End-to-end machine learning system to predict customer churn for a telecom company.

## Live Demo
[Streamlit Dashboard →](YOUR_STREAMLIT_URL)

## Overview
- **Dataset**: IBM Telco Customer Churn — 7,043 customers, 33 features
- **Best Model**: XGBoost — CV AUC 0.9194
- **Churn Rate**: 26.5% (class imbalance handled with SMOTE)

## Pipeline
| Phase | Description |
|---|---|
| EDA | Missing values, distributions, churn patterns |
| Data Cleaning | Drop leakage columns, fix dtypes, handle nulls |
| Feature Engineering | Encoding, scaling, 5 new engineered features |
| Model Training | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Evaluation | ROC-AUC, F1, Precision-Recall, threshold tuning |
| SHAP | Global + local explanations, customer risk scoring |
| Dashboard | Live prediction + risk board via Streamlit |

## Model Results
| Model | CV AUC | CV F1 |
|---|---|---|
| XGBoost | 0.9194 ± 0.0086 | 0.8509 ± 0.0078 |
| LightGBM | 0.9186 ± 0.0084 | 0.8512 ± 0.0089 |
| Logistic Regression | 0.9149 ± 0.0085 | 0.8378 ± 0.0102 |
| Random Forest | 0.9057 ± 0.0079 | 0.8334 ± 0.0084 |

## Tech Stack
`Python` `XGBoost` `LightGBM` `Scikit-learn` `SHAP` `Streamlit` `Pandas` `Matplotlib` `Seaborn`

## Setup
```bash
git clone https://github.com/chinmayee41203/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project Structure
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Cleaned + feature engineered data
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_SHAP.ipynb
├── src/                      # Reusable pipeline scripts
├── models/                   # Trained model artifacts
├── reports/                  # Figures + risk scores
├── dashboard/
│   └── app.py                # Streamlit dashboard
└── requirements.txt
