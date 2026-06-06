"""Project configuration and constants."""

from pathlib import Path

# Dynamically resolve project root
def _get_project_root():
    """Get the project root directory."""
    cwd = Path.cwd()
    if cwd.name.lower() == "notebooks":
        return cwd.parent
    return cwd

PROJECT_ROOT = _get_project_root()

# Dataset Configuration
DATASET_ID = "AmaanP314/youtube-comment-sentiment"

# Data Paths (relative to PROJECT_ROOT)
SAVE_DIR = PROJECT_ROOT / "data" / "youtube-comment-sentiment"
PARQUET_PATH = SAVE_DIR / "train.parquet"
PROCESSED_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "youtube-comments-clean.parquet"

# Model Paths (relative to PROJECT_ROOT)
MODELS_DIR = PROJECT_ROOT / "models"
LOGREG_INFERENCE_MODEL_PATH = MODELS_DIR / "youtube_sentiment_logreg_inference_pipeline"
LOGREG_PIPELINE_MODEL_PATH = MODELS_DIR / "youtube_sentiment_logreg_pipeline"
LABELS_PATH = MODELS_DIR / "youtube_sentiment_labels.json"

# Spark Configuration
SPARK_MASTER = "local[*]"
SPARK_SHUFFLE_PARTITIONS = 8

# ML Pipeline Configuration
ML_VOCAB_SIZE = 10000
ML_MIN_DF = 3
ML_NGRAM_N = 2
ML_MAX_ITER = 50
ML_REG_PARAM = 0.05
ML_ELASTIC_NET_PARAM = 0.0

# Train/Test Split
TRAIN_TEST_SPLIT = [0.8, 0.2]
SAMPLE_FRACTION = 0.40
RANDOM_SEED = 42

#stop words
STOP_WORDS = {
    "the", "and", "for", "you", "that", "this", "with", "are", "was", "but",
    "not", "have", "from", "they", "his", "her", "she", "him", "your",
    "what", "when", "where", "who", "why", "how", "can", "will", "just",
    "about", "like", "get", "all", "out", "one", "its", "it's", "i",
    "a", "an", "to", "of", "in", "on", "is", "it", "as", "at", "be",
    "or", "by", "we", "he", "me", "my", "so", "if", "do", "no", "yes",
    "has", "more", "even", "there", "their", "would", "should",
    "them", "don", "does", "did", "just", "also", "get", "got",
    "make", "made", "much", "many", "still", "really", "very"
}