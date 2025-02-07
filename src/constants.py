import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths ==================================================================
DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data")

# Log Paths ===================================================================
LOG_PATH_MASTER         = os.path.join(DATA_PATH_MASTER, "log")

# Config Paths ================================================================
CONFIG_PATH_MASTER      = os.path.join(PROJECT_PATH, "config")
CONFIG_FILE_FORMAT      = ".yaml"
CONFIG_DEFAULT_FILE     = "default" + CONFIG_FILE_FORMAT
USAGES_FILE             = os.path.join(CONFIG_PATH_MASTER, 
                                       "usages" + CONFIG_FILE_FORMAT)

# Model Paths =================================================================
MODEL_PATH_MASTER       = os.path.join(DATA_PATH_MASTER, "models")
MODEL_PATH_BASE         = os.path.join(MODEL_PATH_MASTER, "base")
MODEL_PATH_FINE_TUNED   = os.path.join(MODEL_PATH_MASTER, "fine-tuned")

# Datasets Paths ==============================================================
DATASETS_PATH_MASTER    = os.path.join(DATA_PATH_MASTER, "datasets")