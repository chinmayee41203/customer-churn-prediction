import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file
RAW_DATA_FILE = RAW_DATA_DIR / "churn_data.csv"

# Model config
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Feature groups (update after EDA)
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = []
DROP_COLUMNS = []

# PostgreSQL (optional, Phase 9)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "database": os.getenv("DB_NAME", "churn_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}