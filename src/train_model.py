"""
Phase 6: Model Training
- Train/test split with stratification
- Handle class imbalance with SMOTE
- Train 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Save all models to models/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'Churn Value'


def load_features(filepath: str = None) -> pd.DataFrame:
    path = filepath or (PROCESSED_DATA_DIR / 'telco_churn_features.csv')
    df = pd.read_csv(path)
    print(f"✅ Loaded features: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Train churn rate: {y_train.mean():.3f} | Test churn rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied: {X_train.shape[0]} → {X_res.shape[0]} samples")
    print(f"   Class balance after SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )
    }


def train_all(X_train, y_train, X_test, y_test):
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        print(f"\n🔄 Training: {name}")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='roc_auc', n_jobs=-1)
        test_score = model.score(X_test, y_test)

        results[name] = {
            'model': model,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_accuracy': test_score
        }

        print(f"   CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Test Acc: {test_score:.4f}")

        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, MODELS_DIR / f'{safe_name}.pkl')
        print(f"   ✅ Saved: models/{safe_name}.pkl")

    return results


def print_leaderboard(results: dict):
    print("\n" + "=" * 55)
    print("MODEL LEADERBOARD (by CV AUC)")
    print("=" * 55)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_auc_mean'], reverse=True)
    for rank, (name, r) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:<25} AUC: {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}  |  Acc: {r['test_accuracy']:.4f}")


def run_training_pipeline(filepath: str = None):
    print("=" * 55)
    print("PHASE 6: MODEL TRAINING")
    print("=" * 55)

    df = load_features(filepath)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    results = train_all(X_train_res, y_train_res, X_test, y_test)
    print_leaderboard(results)

    # Save split for evaluation phase
    joblib.dump((X_train, X_test, y_train, y_test),
                MODELS_DIR / 'train_test_split.pkl')
    print("\n✅ Train/test split saved: models/train_test_split.pkl")
    print("✅ Phase 6 Complete!")
    return results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_training_pipeline()
