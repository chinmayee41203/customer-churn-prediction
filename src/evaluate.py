"""
Phase 7: Model Evaluation
- Confusion matrix, classification report
- ROC-AUC, Precision-Recall curves
- Threshold tuning
- Feature importance
- Full model comparison table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay
)
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / 'train_test_split.pkl')
    models = {
        'XGBoost'            : joblib.load(MODELS_DIR / 'xgboost.pkl'),
        'LightGBM'           : joblib.load(MODELS_DIR / 'lightgbm.pkl'),
        'Logistic Regression': joblib.load(MODELS_DIR / 'logistic_regression.pkl'),
        'Random Forest'      : joblib.load(MODELS_DIR / 'random_forest.pkl'),
    }
    print(f"✅ Loaded {len(models)} models + test split ({X_test.shape[0]} samples)")
    return models, X_test, y_test


def evaluate_best(model, X_test, y_test, name='XGBoost'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"EVALUATION — {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'confusion_matrix_{name.lower().replace(" ","_")}.png')
    plt.close()
    return y_pred, y_prob


def plot_roc_all(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#C44E52', '#4C72B0', '#55A868', '#DD8452']
    for (name, model), color in zip(models.items(), colors):
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curves_all_models.png')
    plt.close()
    print("✅ ROC curve saved")


def tune_threshold(y_test, y_prob):
    results = []
    for t in np.arange(0.2, 0.7, 0.05):
        pred = (y_prob >= t).astype(int)
        results.append({
            'Threshold': round(t, 2),
            'Precision': round(precision_score(y_test, pred), 4),
            'Recall'   : round(recall_score(y_test, pred), 4),
            'F1'       : round(f1_score(y_test, pred), 4),
        })
    df = pd.DataFrame(results)
    best = df.loc[df['F1'].idxmax(), 'Threshold']
    print(f"\n✅ Best threshold by F1: {best}")
    print(df.to_string(index=False))
    return df, best


def model_comparison_table(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        rows.append({
            'Model'    : name,
            'ROC-AUC'  : round(roc_auc_score(y_test, prob), 4),
            'F1'       : round(f1_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred), 4),
            'Recall'   : round(recall_score(y_test, pred), 4),
        })
    df = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
    print("\n✅ Model comparison saved to reports/model_comparison.csv")
    print(df.to_string(index=False))
    return df


def run_evaluation():
    print("=" * 50)
    print("PHASE 7: MODEL EVALUATION")
    print("=" * 50)
    models, X_test, y_test = load_artifacts()
    best = models['XGBoost']
    y_pred, y_prob = evaluate_best(best, X_test, y_test)
    plot_roc_all(models, X_test, y_test)
    tune_threshold(y_test, y_prob)
    model_comparison_table(models, X_test, y_test)
    print("\n✅ Phase 7 Complete!")


if __name__ == "__main__":
    run_evaluation()
