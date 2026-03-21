"""Constants and settings for the Kaggle Dataset Analyzer."""

MAX_FILE_SIZE_MB = 100
LARGE_FILE_WARNING_MB = 50
PREVIEW_ROWS = 10
MAX_HEATMAP_COLS = 20
PAIRPLOT_SAMPLE_SIZE = 1000
MAX_PIE_CATEGORIES = 10
MISSING_DROP_ROW_THRESHOLD = 0.20   # drop rows if col has <20% missing
MISSING_DROP_COL_THRESHOLD = 0.50   # drop col if >50% missing
CATEGORY_UNIQUE_THRESHOLD = 0.50    # convert to category if <50% unique

COLORS = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "error": "#d62728",
    "neutral": "#7f7f7f",
}

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

