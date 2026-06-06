from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    HashingTF,
    IDF,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    LinearSVC,
    GBTClassifier,
    DecisionTreeClassifier,
    NaiveBayes,
    OneVsRest,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(spark: SparkSession, parquet_path: str) -> DataFrame:
    """Read a parquet file and return the Spark DataFrame."""
    return spark.read.parquet(parquet_path)


def summarise_counts(raw_path: str, clean_df: DataFrame, spark: SparkSession) -> None:
    """Print row counts before and after RDD preprocessing."""
    raw_count = spark.read.parquet(raw_path).count()
    clean_count = clean_df.count()
    print(f"Original comments:                {raw_count}")
    print(f"Comments after RDD preprocessing: {clean_count}")
    print(f"Removed comments:                 {raw_count - clean_count}")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

MODEL_COLUMNS = [
    "CommentText",
    "clean_text",
    "Sentiment",
    "Likes",
    "Replies",
    "comment_word_count",
    "engagement_total",
    "engagement_level",
]


def select_model_columns(df: DataFrame, columns: list[str] | None = None) -> DataFrame:
    """Select only the columns needed for modelling."""
    cols = columns or MODEL_COLUMNS
    return df.select(*cols)


# ---------------------------------------------------------------------------
# Sampling & splitting
# ---------------------------------------------------------------------------

def sample_dataset(df: DataFrame, fraction: float = 0.40, seed: int = 42) -> DataFrame:
    """Return a random sample, cached for reuse."""
    return df.sample(withReplacement=False, fraction=fraction, seed=seed).cache()


def train_test_split(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """Split df into (train, test) DataFrames, both cached."""
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return train_df.cache(), test_df.cache()


# ---------------------------------------------------------------------------
# Shared feature extraction stages (reused by every pipeline)
# ---------------------------------------------------------------------------

def _feature_stages(vocab_size: int = 10_000, min_df: int = 3, ngram_n: int = 2):
    """Return the list of feature-engineering stages shared by all pipelines."""
    tokenizer  = Tokenizer(inputCol="clean_text", outputCol="words")
    remover    = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    tf_uni  = HashingTF(inputCol="filtered_words", outputCol="tf_uni", numFeatures=vocab_size)
    idf_uni = IDF(inputCol="tf_uni", outputCol="idf_uni", minDocFreq=min_df)

    bigram  = NGram(n=ngram_n, inputCol="filtered_words", outputCol="bigrams")
    tf_bi   = HashingTF(inputCol="bigrams", outputCol="tf_bi", numFeatures=vocab_size)
    idf_bi  = IDF(inputCol="tf_bi", outputCol="idf_bi", minDocFreq=min_df)

    assembler    = VectorAssembler(inputCols=["idf_uni", "idf_bi"], outputCol="features")
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label", handleInvalid="keep")

    return [tokenizer, remover, tf_uni, idf_uni, bigram, tf_bi, idf_bi, assembler, label_indexer]


def build_pipeline(classifier, vocab_size: int = 10_000, min_df: int = 3, ngram_n: int = 2) -> Pipeline:
    """
    Build a full TF-IDF + classifier Pipeline for any pyspark.ml classifier.

    Parameters
    ----------
    classifier : pyspark.ml classifier instance
        Already configured (e.g. LogisticRegression(...)).  Use OneVsRest
        to wrap binary-only classifiers (LinearSVC, GBTClassifier).
    vocab_size, min_df, ngram_n : feature hyperparameters (see _feature_stages).
    """
    stages = _feature_stages(vocab_size=vocab_size, min_df=min_df, ngram_n=ngram_n)
    return Pipeline(stages=stages + [classifier])


# ---------------------------------------------------------------------------
# Ready-made pipeline builders (convenience wrappers)
# ---------------------------------------------------------------------------

def build_lr_pipeline(vocab_size=10_000, min_df=3, ngram_n=2,
                      max_iter=50, reg_param=0.05, elastic_net=0.0) -> Pipeline:
    clf = LogisticRegression(
        featuresCol="features", labelCol="label",
        maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net,
    )
    return build_pipeline(clf, vocab_size, min_df, ngram_n)


def build_rf_pipeline(vocab_size=10_000, min_df=3, ngram_n=2,
                      num_trees=100, max_depth=10, seed=42) -> Pipeline:
    clf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=num_trees, maxDepth=max_depth, seed=seed,
    )
    return build_pipeline(clf, vocab_size, min_df, ngram_n)


def build_svm_pipeline(vocab_size=10_000, min_df=3, ngram_n=2,
                       max_iter=100, reg_param=0.01) -> Pipeline:
    svm = LinearSVC(featuresCol="features", labelCol="label",
                    maxIter=max_iter, regParam=reg_param)
    ovr = OneVsRest(classifier=svm, featuresCol="features", labelCol="label")
    return build_pipeline(ovr, vocab_size, min_df, ngram_n)


def build_dt_pipeline(vocab_size=10_000, min_df=3, ngram_n=2,
                      max_depth=10, seed=42) -> Pipeline:
    clf = DecisionTreeClassifier(
        featuresCol="features", labelCol="label",
        maxDepth=max_depth, seed=seed,
    )
    return build_pipeline(clf, vocab_size, min_df, ngram_n)


def build_nb_pipeline(vocab_size=10_000, min_df=3, ngram_n=2,
                      smoothing=1.0) -> Pipeline:
    """Naive Bayes — requires non-negative features; TF-IDF satisfies this."""
    clf = NaiveBayes(featuresCol="features", labelCol="label", smoothing=smoothing)
    return build_pipeline(clf, vocab_size, min_df, ngram_n)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

METRICS = ("accuracy", "f1", "weightedPrecision", "weightedRecall")


def evaluate_model(
    predictions: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> dict[str, float]:
    """Return {metric_name: value} for the standard classification metrics."""
    results: dict[str, float] = {}
    for metric in METRICS:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=prediction_col, metricName=metric,
        )
        results[metric] = evaluator.evaluate(predictions)
    return results


def print_evaluation_metrics(metrics: dict[str, float], model_name: str = "") -> None:
    """Pretty-print the evaluation metrics dict."""
    if model_name:
        print(f"\n{'='*40}\n{model_name}\n{'='*40}")
    labels = {
        "accuracy":          "Accuracy ",
        "f1":                "F1-score ",
        "weightedPrecision": "Precision",
        "weightedRecall":    "Recall   ",
    }
    for key, label in labels.items():
        if key in metrics:
            print(f"  {label}: {metrics[key]:.4f}")


def plot_confusion_matrix(
    predictions: DataFrame,
    class_labels: list[str],
    model_name: str = "Model",
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> None:
    """
    Compute and plot a normalised confusion matrix from a predictions DataFrame.

    Parameters
    ----------
    predictions   : DataFrame with label and prediction columns.
    class_labels  : Ordered list of class names matching the StringIndexer order.
    model_name    : Title shown above the plot.
    """
    # Collect (label, prediction) pairs — small after aggregation
    pairs = (
        predictions
        .select(label_col, prediction_col)
        .groupBy(label_col, prediction_col)
        .count()
        .collect()
    )

    n = len(class_labels)
    matrix = np.zeros((n, n), dtype=int)
    for row in pairs:
        true_idx = int(row[label_col])
        pred_idx = int(row[prediction_col])
        if true_idx < n and pred_idx < n:
            matrix[true_idx][pred_idx] += row["count"]

    # Normalise rows to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(xmax=1))

    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_labels, yticklabels=class_labels,
        xlabel="Predicted label", ylabel="True label",
        title=f"Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(n):
        for j in range(n):
            colour = "white" if norm_matrix[i, j] > 0.55 else "black"
            ax.text(j, i, f"{norm_matrix[i,j]:.1%}\n({matrix[i,j]:,})",
                    ha="center", va="center", fontsize=9, color=colour)

    plt.tight_layout()
    plt.show()


def compare_models(results: dict[str, dict[str, float]]) -> None:
    """
    Bar chart comparing all models across the four metrics.

    Parameters
    ----------
    results : {model_name: metrics_dict} as returned by evaluate_model.
    """
    metric_labels = {
        "accuracy": "Accuracy", "f1": "F1-score",
        "weightedPrecision": "Precision", "weightedRecall": "Recall",
    }
    model_names = list(results.keys())
    x = np.arange(len(metric_labels))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metric_labels]
        bars = ax.bar(x + i * width, values, width, label=name)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels.values())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: PipelineModel, path: str) -> None:
    """Save a fitted PipelineModel, overwriting any existing file."""
    model.write().overwrite().save(path)
    print(f"Model saved at: {path}")