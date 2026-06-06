"""
Phase 8: Explainable AI — SHAP
- Global feature importance (summary + bar plots)
- Individual prediction explanations (waterfall)
- Dependence plots for key features
- High-risk customer identification with risk tiers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = joblib.load(MODELS_DIR / 'xgboost.pkl')
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / 'train_test_split.pkl')
    print(f"✅ Loaded XGBoost + test set ({X_test.shape[0]} samples)")
    return model, X_test, y_test


def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    joblib.dump(explainer, MODELS_DIR / 'shap_explainer.pkl')
    print(f"✅ SHAP values computed: {shap_values.shape}")
    return explainer, shap_values


def plot_summary(shap_values, X_test):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title('SHAP Summary Plot — XGBoost', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_summary_beeswarm.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, max_display=15)
    plt.title('SHAP Feature Importance — Mean |SHAP|', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_bar_importance.png', bbox_inches='tight')
    plt.close()
    print("✅ Summary plots saved")


def plot_waterfall(explainer, shap_values, X_test, y_test, idx=None, label='churned'):
    if idx is None:
        indices = np.where(y_test.values == (1 if label == 'churned' else 0))[0]
        idx = indices[0]
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx],
        feature_names=X_test.columns.tolist()
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(f'SHAP Waterfall — {label.capitalize()} Customer')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'shap_waterfall_{label}.png', bbox_inches='tight')
    plt.close()
    print(f"✅ Waterfall plot saved for {label} customer")


def build_risk_scores(model, shap_values, X_test, y_test):
    churn_prob = model.predict_proba(X_test)[:, 1]
    risk_df = X_test.copy()
    risk_df['Churn Probability'] = churn_prob.round(4)
    risk_df['SHAP Magnitude'] = np.abs(shap_values).mean(axis=1).round(4)
    risk_df['Actual Churn'] = y_test.values
    risk_df['Risk Tier'] = pd.cut(
        risk_df['Churn Probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    risk_df.to_csv(REPORTS_DIR / 'customer_risk_scores.csv', index=False)
    print(f"\n✅ Risk scores saved to reports/customer_risk_scores.csv")
    print(f"Risk Tier Distribution:\n{risk_df['Risk Tier'].value_counts()}")
    return risk_df


def run_shap_pipeline():
    print("=" * 50)
    print("PHASE 8: SHAP EXPLAINABILITY")
    print("=" * 50)
    model, X_test, y_test = load_artifacts()
    explainer, shap_values = compute_shap(model, X_test)
    plot_summary(shap_values, X_test)
    plot_waterfall(explainer, shap_values, X_test, y_test, label='churned')
    plot_waterfall(explainer, shap_values, X_test, y_test, label='retained')

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.to_csv(REPORTS_DIR / 'shap_values.csv', index=False)

    build_risk_scores(model, shap_values, X_test, y_test)
    print("\n✅ Phase 8 Complete!")


if __name__ == "__main__":
    run_shap_pipeline()
