# Big Data Analytics Project

End-to-end big data project for sentiment analytics, combining:

- PySpark data engineering and classical ML
- Transformer-based deep learning experiments
- Graph analytics on DBLP citation data
- Near real-time streaming inference with Kafka + MongoDB over Tailscale

The sentiment dataset used in the core pipeline is:

- Hugging Face: https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment

We also used for graph analysis: https://www.kaggle.com/datasets/mathurinache/citation-network-dataset

## What This Repository Contains

This repository is organized as a notebook-driven workflow backed by reusable utilities in src and a containerized infra stack in infra.

Main capabilities:

- Ingest and preprocess large-scale YouTube comment data with Spark
- Train and compare multiple Spark MLlib-compatible models
- Fine-tune DistilBERT for 3-class sentiment classification
- Run graph preprocessing and graph algorithms on large DBLP citation data
- Stream live YouTube comments into Kafka, score them in Spark Structured Streaming, and persist results to MongoDB

## Repository Structure

```text
.
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── Project - BigDataAnalaysis - 2025.pdf
│   └── powerpoint_bda.pptx
├── infra/
│   ├── .env.example
│   ├── docker-compose.yml
│   ├── README.md
│   └── youtube_producer/
│       ├── Dockerfile
│       ├── producer.py
│       └── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ml_classification.ipynb
│   ├── 03_dl_classification.ipynb
│   ├── 04_graph_preprocessing.ipynb
│   ├── 05_graph_analytics.ipynb
│   └── 06_streaming.ipynb
├── results/
│   └── model_comparison.csv
└── src/
    ├── config.py
    ├── ml_pipeline.py
    ├── utils.py
    ├── utils_dl.py
    └── utils_ml.py
```

## Notebook Workflow

### 01_eda.ipynb

- Loads and explores the YouTube sentiment dataset in Spark
- Performs cleaning and quality checks
- Produces processed outputs used by downstream notebooks

### 02_ml_classification.ipynb

- Trains and evaluates classical Spark pipelines on shared features/splits
- Compares models with common metrics and confusion matrices
- The streaming notebook uses the best-performing model selected here (Linear SVM OvR pipeline)

### 03_dl_classification.ipynb

- Fine-tunes DistilBERT via `TorchDistributor` for distributed PyTorch training
- Uses `predict_batch_udf` for Spark-native distributed inference
- Evaluates with `MulticlassClassificationEvaluator` and a per-class Spark SQL breakdown

### 04_graph_preprocessing.ipynb

- Preprocesses DBLP V12 citation network into graph-ready Parquet
- Converts large JSON array to JSONL for scalable Spark ingestion

### 05_graph_analytics.ipynb

- Runs GraphFrames algorithms on the prepared DBLP graph
- Includes full-graph and focused subgraph analyses

### 06_streaming.ipynb

- Connects to Kafka and MongoDB
- Loads the selected Spark pipeline model
- Scores streaming comments via micro-batch processing
- Upserts predictions into MongoDB and inspects output distributions

Important streaming components:

- Producer path: infra/youtube_producer/producer.py
- Kafka topic: youtube_comments
- MongoDB write strategy: foreachBatch + bulk upsert

## Core Python Modules (src)

- src/config.py
	- Centralized paths, Spark defaults, ML hyperparameters, dataset IDs

- src/utils.py
	- Spark session bootstrap and data cleaning/record preparation helpers

- src/utils_ml.py
	- Reusable Spark ML training/evaluation utilities and model builders
	- Includes LR, SVM OvR, NB, DT, RF pipelines

- src/ml_pipeline.py
	- Standalone configurable Spark ML pipeline assembly and metric evaluation helpers

- src/utils_dl.py
	- Spark-to-HuggingFace data preparation
	- DistilBERT training/inference helpers for PyTorch and TensorFlow/Keras

## Streaming Infrastructure (infra)

The infra folder contains a Docker Compose stack with:

- tailscale: secure tailnet exposure
- mongodb: checkpoint and scored-result persistence
- kafka: broker for comment stream transport
- youtube-producer: polls YouTube API and publishes normalized messages

Producer behavior summary:

- Maintains a per-video checkpoint in MongoDB
- First run backfills historical comments
- Later runs stream only new comments after checkpoint
- Publishes normalized JSON payloads keyed by comment_id

Message schema:

```json
{
	"comment_id": "string",
	"video_id": "string",
	"text": "string",
	"author": "string",
	"published_at": "2026-06-02T12:34:56Z",
	"fetched_at": "2026-06-02T12:35:01Z",
	"source": "youtube_api"
}
```

## Environment Setup

### Prerequisites

- Python 3.11+ recommended
- Java runtime compatible with Spark
- Docker + Docker Compose (for infra stack)
- Tailscale account and auth key (for infra networking)

### Local Python Environment

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Infra Environment

Inside infra:

```bash
cp .env.example .env
```

Set at minimum:

- TS_AUTHKEY
- TS_IP_ADDRESS
- YOUTUBE_API_KEY
- YOUTUBE_VIDEO_ID
- MONGO_USER
- MONGO_PASS

Then start infra:

```bash
cd infra
docker compose up -d --build
```

## How To Run

Typical order:

1. Run 01_eda.ipynb to ingest/clean/source processed data.
2. Run 02_ml_classification.ipynb to compare models.
3. Run 03_dl_classification.ipynb for DistilBERT experiments.
4. Optional graph workflow:
	 - Run 04_graph_preprocessing.ipynb
	 - Run 05_graph_analytics.ipynb
5. Start infra stack and run 06_streaming.ipynb for live scoring.

## Results and Artifacts

- results/model_comparison.csv
  - Stores comparative metrics from classical ML runs

Runtime outputs (not tracked in Git) are generated under data/ and models/ when running notebooks and streaming jobs.

## Security and Networking Notes

- TS_IP_ADDRESS is network location metadata, not an application secret by itself.
- Access is gated by tailnet authentication and visibility rules.
- Real credentials/secrets (API keys, auth keys, passwords) must remain in env files/secrets management.

## Known Limitations

- Classical Spark ML pipelines can underperform on multilingual or nuanced short text.
- Example observed in streaming analysis: a short Portuguese negative comment like "Lixo" can be misclassified as Neutral.
- MLlib-first streaming inference improves operational compatibility and latency, but may trade off absolute accuracy versus stronger transformer models.

## Future Improvements

- Integrate a stronger multilingual transformer for higher semantic accuracy.
- Add model/version registry and automated streaming quality monitoring.
- Add reproducible CLI/Makefile tasks for all notebook stages.
- Expand README and docs with benchmark tables tied to named models in results/model_comparison.csv.
