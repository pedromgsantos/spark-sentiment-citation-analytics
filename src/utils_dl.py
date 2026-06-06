"""
utils_dl.py
-----------
Deep Learning helpers for YouTube sentiment classification.

Pipeline philosophy
~~~~~~~~~~~~~~~~~~~
Spark handles everything up to and including the train/val/test split and
label encoding — all written back to Parquet.  The HuggingFace `datasets`
library then loads those Parquet files directly into Arrow-backed Dataset
objects, so no pandas conversion is ever required.

    Spark  →  clean + encode labels + stratified split  →  Parquet
    HuggingFace datasets  →  tokenise (cached, batched)  →  DataLoader / tf.data
    PyTorch / Keras  →  fine-tune DistilBERT

Exported names (imported by the notebook)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data preparation (Spark-side)
    spark_prepare_and_split

HuggingFace loading + tokenisation
    load_hf_splits, tokenize_splits, get_tokenizer

PyTorch
    build_pytorch_model, train_pytorch, predict_pytorch

Keras / TF
    build_keras_model, train_keras, predict_keras

Evaluation & visualisation
    print_classification_report, plot_confusion_matrix,
    plot_training_history, compare_models, LABEL_NAMES
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── HuggingFace ────────────────────────────────────────────────────────────
PRETRAINED  = "distilbert-base-uncased"
MAX_LEN     = 128
LABEL_NAMES = ["Negative", "Neutral", "Positive"]   # fixed order → 0, 1, 2
LABEL2ID    = {name: idx for idx, name in enumerate(LABEL_NAMES)}


# ---------------------------------------------------------------------------
# 1.  Spark-side: clean, encode labels, stratified split, write Parquet
# ---------------------------------------------------------------------------

def spark_prepare_and_split(
    spark_df,
    output_dir: str,
    text_col:   str   = "clean_text",
    label_col:  str   = "Sentiment",
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int          = 42,
) -> None:
    """
    Full Spark-side preparation pipeline:
      1. Select text + label, drop nulls/empty rows.
      2. Encode string labels to integer ``label_id`` using a MapType lookup
         (no UDF, no pandas) so the column lands in the Parquet schema.
      3. Stratified split via per-class ``randomSplit`` — true stratification,
         not an approximation.
      4. Write train / val / test splits to ``output_dir/{split}/``.

    The written Parquet files contain exactly two columns:
        text      string
        label_id  int   (0=Negative, 1=Neutral, 2=Positive)

    Parameters
    ----------
    spark_df    : pyspark.sql.DataFrame  — must contain text_col and label_col
    output_dir  : local or HDFS/S3 path  — e.g. "../data/deep_learning"
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType

    # ── 1. Select & clean ──────────────────────────────────────────────────
    df = (
        spark_df
        .select(
            F.col(text_col).alias("text"),
            F.col(label_col).alias("label"),
        )
        .dropna(subset=["text", "label"])
        .filter(F.trim(F.col("text")) != "")
    )

    # ── 2. Encode labels (map lookup — no pandas, no UDF) ──────────────────
    # Build a small mapping DataFrame and broadcast-join it.
    spark = df.sparkSession
    mapping_rows = [(name, idx) for name, idx in LABEL2ID.items()]
    mapping_df   = spark.createDataFrame(mapping_rows, schema=["label", "label_id"])
    mapping_df   = mapping_df.withColumn("label_id",
                                         F.col("label_id").cast(IntegerType()))

    df = (
        df.join(F.broadcast(mapping_df), on="label", how="inner")
          .select("text", "label_id")
    )

    # ── 3. True stratified split ───────────────────────────────────────────
    test_ratio      = 1.0 - train_ratio - val_ratio
    val_ratio_inner = val_ratio / (train_ratio + val_ratio)   # within non-test

    train_parts, val_parts, test_parts = [], [], []

    for label_val in LABEL2ID.values():
        class_df = df.filter(F.col("label_id") == label_val)
        # First cut: separate test
        non_test, test_split = class_df.randomSplit(
            [train_ratio + val_ratio, test_ratio], seed=seed
        )
        # Second cut: train vs val from the non-test portion
        train_split, val_split = non_test.randomSplit(
            [1.0 - val_ratio_inner, val_ratio_inner], seed=seed
        )
        train_parts.append(train_split)
        val_parts.append(val_split)
        test_parts.append(test_split)

    def _union(parts):
        result = parts[0]
        for part in parts[1:]:
            result = result.union(part)
        return result

    train_df = _union(train_parts).orderBy(F.rand(seed=seed))
    val_df   = _union(val_parts).orderBy(F.rand(seed=seed))
    test_df  = _union(test_parts).orderBy(F.rand(seed=seed))

    # ── 4. Write Parquet ───────────────────────────────────────────────────
    train_df.write.parquet(f"{output_dir}/train", mode="overwrite")
    val_df.write.parquet(f"{output_dir}/val",     mode="overwrite")
    test_df.write.parquet(f"{output_dir}/test",   mode="overwrite")

    # Print counts for sanity check
    print(f"[spark_prepare_and_split] written to {output_dir}")
    print(f"  train : {train_df.count():>7,}")
    print(f"  val   : {val_df.count():>7,}")
    print(f"  test  : {test_df.count():>7,}")

# ---------------------------------------------------------------------------
# 5.  Evaluation & visualisation (shared)
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Print sklearn classification report; return metrics dict."""
    print(f"\n{'='*45}\n{model_name}\n{'='*45}")
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
    )
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    return {
        "accuracy":          report["accuracy"],
        "f1":                report["weighted avg"]["f1-score"],
        "weightedPrecision": report["weighted avg"]["precision"],
        "weightedRecall":    report["weighted avg"]["recall"],
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Normalised confusion matrix heatmap."""
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        norm, annot=False, cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        vmin=0, vmax=1, ax=ax,
    )
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            colour = "white" if norm[i, j] > 0.55 else "black"
            ax.text(
                j + 0.5, i + 0.5,
                f"{norm[i,j]:.1%}\n({cm[i,j]:,})",
                ha="center", va="center", fontsize=9, color=colour,
            )
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, model_name: str = "Model") -> None:
    """
    Loss + accuracy curves.
    Accepts a Keras History object or a plain dict {train_loss, val_loss, val_acc}.
    """
    if hasattr(history, "history"):
        train_loss = history.history["loss"]
        val_loss   = history.history["val_loss"]
        val_acc    = history.history["val_accuracy"]
    else:
        train_loss = history["train_loss"]
        val_loss   = history["val_loss"]
        val_acc    = history["val_acc"]

    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, "o-",  label="Train loss")
    ax1.plot(epochs, val_loss,   "s--", label="Val loss")
    ax1.set(title=f"{model_name} — Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend()

    ax2.plot(epochs, val_acc, "^-", color="green", label="Val accuracy")
    ax2.set(title=f"{model_name} — Val Accuracy",
            xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    ax2.legend()

    plt.suptitle(model_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
