import os
import re
import sys
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


def get_project_root():
    """
    Get the project root directory.
    Works correctly whether called from notebooks or scripts.
    
    Returns:
        Path: Absolute path to project root.
    """
    cwd = Path.cwd()
    
    # If we're in the notebooks directory, go up one level
    if cwd.name.lower() == "notebooks":
        return cwd.parent
    
    # Otherwise, assume we're already at project root
    return cwd


def create_spark_session(app_name="spark-app", shuffle_partitions=8):
    """
    Create and configure a Spark session.
    
    Args:
        app_name (str): The application name for Spark.
        shuffle_partitions (int): Number of partitions for shuffle operations.
    
    Returns:
        SparkSession: Configured Spark session.
    """
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.driver.bindAddress", "127.0.0.1")  # aaaa
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )

    spark.sparkContext.addPyFile(str(Path(__file__).resolve()))
    
    return spark


def clean_text_column(df, input_col="text", output_col="clean_text"):
    cleaned_df = (
        df
        .withColumn(output_col, F.lower(F.col(input_col)))
        .withColumn(output_col, F.regexp_replace(output_col, r"https?://\S+|www\.\S+", " "))
        .withColumn(output_col, F.regexp_replace(output_col, r"@\w+", " "))
        .withColumn(output_col, F.regexp_replace(output_col, r"[^\p{L}\p{N}\s]", " "))
        .withColumn(output_col, F.regexp_replace(output_col, r"\s+", " "))
        .withColumn(output_col, F.trim(output_col))
        .filter(F.col(output_col).isNotNull())
        .filter(F.length(F.col(output_col)) > 0)
    )

    return cleaned_df


def clean_comment_text(comment_text):
    if comment_text is None:
        return ""

    text = str(comment_text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_integer(value):
    if value is None:
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def clean_country_code(country_code):
    if country_code is None:
        return "Unknown"

    country = str(country_code).strip().upper()
    if country == "":
        return "Unknown"

    return country[:2]


def classify_engagement(total_engagement):
    if total_engagement >= 20:
        return "high"
    if total_engagement >= 5:
        return "medium"
    if total_engagement > 0:
        return "low"
    return "none"


def prepare_comment_record(row):
    record = row.asDict(recursive=True)

    clean_text = clean_comment_text(record.get("CommentText"))
    likes = safe_integer(record.get("Likes"))
    replies = safe_integer(record.get("Replies"))
    engagement_total = likes + replies

    record["Likes"] = likes
    record["Replies"] = replies
    record["CountryCode"] = clean_country_code(record.get("CountryCode"))
    record["clean_text"] = clean_text
    record["comment_word_count"] = len(clean_text.split()) if clean_text else 0
    record["comment_char_count"] = len(clean_text)
    record["engagement_total"] = engagement_total
    record["engagement_level"] = classify_engagement(engagement_total)

    return Row(**record)


def build_clean_comments_schema(raw_schema):
    existing_fields = []

    for field in raw_schema.fields:
        if field.name in {"Likes", "Replies"}:
            existing_fields.append(StructField(field.name, IntegerType(), nullable=False))
        elif field.name == "CountryCode":
            existing_fields.append(StructField(field.name, StringType(), nullable=False))
        else:
            existing_fields.append(field)

    return StructType(existing_fields + [
        StructField("clean_text", StringType(), nullable=False),
        StructField("comment_word_count", IntegerType(), nullable=False),
        StructField("comment_char_count", IntegerType(), nullable=False),
        StructField("engagement_total", IntegerType(), nullable=False),
        StructField("engagement_level", StringType(), nullable=False),
    ])
