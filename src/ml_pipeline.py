"""Machine Learning pipeline utilities."""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer,
    NGram,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class MLPipelineConfig:
    """Configuration for ML pipeline stages."""
    
    def __init__(
        self,
        vocab_size=10000,
        min_df=3,
        ngram_n=2,
        max_iter=50,
        reg_param=0.05,
        elastic_net_param=0.0
    ):
        """
        Initialize ML pipeline configuration.
        
        Args:
            vocab_size (int): Vocabulary size for CountVectorizer.
            min_df (int): Minimum document frequency for CountVectorizer.
            ngram_n (int): N-gram size.
            max_iter (int): Maximum iterations for LogisticRegression.
            reg_param (float): Regularization parameter.
            elastic_net_param (float): ElasticNet mixing parameter.
        """
        self.vocab_size = vocab_size
        self.min_df = min_df
        self.ngram_n = ngram_n
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.elastic_net_param = elastic_net_param


def build_ml_pipeline(config=None):
    """
    Build a complete ML pipeline for text classification with unigrams and bigrams.
    
    Args:
        config (MLPipelineConfig, optional): Pipeline configuration. 
                                            Defaults to None (uses default values).
    
    Returns:
        Pipeline: A Spark ML pipeline with all stages.
    """
    if config is None:
        config = MLPipelineConfig()
    
    # Tokenize text
    tokenizer = RegexTokenizer(
        inputCol="clean_text",
        outputCol="words",
        pattern="\\s+"
    )
    
    # Remove common stopwords
    stop_words_remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )
    
    # Create bigrams
    ngram = NGram(
        n=config.ngram_n,
        inputCol="filtered_words",
        outputCol="bigrams"
    )
    
    # Unigram CountVectorizer
    unigram_vectorizer = CountVectorizer(
        inputCol="filtered_words",
        outputCol="raw_unigram_features",
        vocabSize=config.vocab_size,
        minDF=config.min_df
    )
    
    # Bigram CountVectorizer
    bigram_vectorizer = CountVectorizer(
        inputCol="bigrams",
        outputCol="raw_bigram_features",
        vocabSize=config.vocab_size,
        minDF=config.min_df
    )
    
    # IDF for unigrams
    unigram_idf = IDF(
        inputCol="raw_unigram_features",
        outputCol="unigram_features"
    )
    
    # IDF for bigrams
    bigram_idf = IDF(
        inputCol="raw_bigram_features",
        outputCol="bigram_features"
    )
    
    # Combine unigram and bigram features
    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"],
        outputCol="features"
    )
    
    # Convert text labels into numbers
    label_indexer = StringIndexer(
        inputCol="Sentiment",
        outputCol="label",
        handleInvalid="skip"
    )
    
    # Logistic Regression classifier
    lr_classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=config.max_iter,
        regParam=config.reg_param,
        elasticNetParam=config.elastic_net_param
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[
        tokenizer,
        stop_words_remover,
        ngram,
        unigram_vectorizer,
        bigram_vectorizer,
        unigram_idf,
        bigram_idf,
        assembler,
        label_indexer,
        lr_classifier
    ])
    
    return pipeline


def evaluate_model(predictions, metrics_list=None):
    """
    Evaluate model predictions using multiple metrics.
    
    Args:
        predictions (DataFrame): DataFrame with predictions and labels.
        metrics_list (list, optional): List of metrics to compute. 
                                      Defaults to ['accuracy', 'f1', 'weightedPrecision', 'weightedRecall'].
    
    Returns:
        dict: Dictionary with metric names and their values.
    """
    if metrics_list is None:
        metrics_list = ['accuracy', 'f1', 'weightedPrecision', 'weightedRecall']
    
    results = {}
    
    for metric_name in metrics_list:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName=metric_name
        )
        results[metric_name] = evaluator.evaluate(predictions)
    
    return results


def print_evaluation_metrics(metrics_dict):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics_dict (dict): Dictionary of metrics from evaluate_model().
    """
    metric_names_display = {
        'accuracy': 'Accuracy',
        'f1': 'F1-score',
        'weightedPrecision': 'Precision',
        'weightedRecall': 'Recall'
    }
    
    for metric_key, metric_value in metrics_dict.items():
        display_name = metric_names_display.get(metric_key, metric_key)
        print(f"{display_name}: {metric_value:.4f}")
