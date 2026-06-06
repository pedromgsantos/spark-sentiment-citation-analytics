# Spark Sentiment & Citation Analytics

Project for the Big Data Analysis course, Master's in Data Science and Advanced Analytics, NOVA IMS.

This repository is organized as a notebook-driven workflow backed by reusable utilities in `src/`
and a containerized streaming stack in `infra/`.

This repository reflects a cleaned version of the project, with improved organization and documentation for clarity and reproducibility.

---

## Project Authors

Pedro Santos   – 20XXXXXX – [20XXXXXX@novaims.unl.pt](mailto:20XXXXXX@novaims.unl.pt)
[Teammate 2]   – 20XXXXXX – [20XXXXXX@novaims.unl.pt](mailto:20XXXXXX@novaims.unl.pt)
[Teammate 3]   – 20XXXXXX – [20XXXXXX@novaims.unl.pt](mailto:20XXXXXX@novaims.unl.pt)
[Teammate 4]   – 20XXXXXX – [20XXXXXX@novaims.unl.pt](mailto:20XXXXXX@novaims.unl.pt)

---

## Project Overview

YouTube generates billions of comments daily, and the DBLP citation network spans millions of
academic publications. Both represent real-world big data challenges requiring scalable tooling.

This project builds an end-to-end Apache Spark pipeline that classifies YouTube comment sentiment
at scale, and mines the DBLP citation network to map research community structure and field
evolution over time.

Datasets used:
- [YouTube Comment Sentiment](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment) – HuggingFace
- [DBLP Citation Network V12](https://www.kaggle.com/datasets/mathurinache/citation-network-dataset) – Kaggle

---

## Project Goals

1. **Data Engineering** – Ingest, clean, and transform both datasets using scalable PySpark
   DataFrames and SparkSQL, with schema-on-read and partition-aware pipelines.
2. **ML Classification** – Train and compare multiple Spark MLlib models for 3-class sentiment
   classification, selecting the best pipeline for downstream streaming use.
3. **Deep Learning** – Fine-tune DistilBERT using distributed PyTorch via TorchDistributor,
   with Spark-native inference and evaluation.
4. **Graph Analytics** – Preprocess the DBLP citation network and run GraphFrames algorithms
   (PageRank, connected components, community detection, motif finding) to extract research
   community insights.
5. **Streaming** – Deploy a live inference pipeline that ingests YouTube comments from Kafka,
   scores them with the selected MLlib model, and persists results to MongoDB.
6. **Stakeholder Communication** – Deliver management-facing findings framed around the 4 V's
   of Big Data, with no algorithm or hyperparameter detail in the presentation.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── docs/                        # Project brief and presentation
│   ├── Project - BigDataAnalysis - 2025.pdf
│   └── powerpoint_bda.pptx
├── infra/                       # Docker Compose streaming stack
│   ├── .env.example
│   ├── docker-compose.yml
│   ├── README.md
│   └── youtube_producer/        # YouTube API -> Kafka producer
│       ├── Dockerfile
│       ├── producer.py
│       └── requirements.txt
├── notebooks/                   # Main workflow (run in order)
│   ├── 01_eda.ipynb
│   ├── 02_ml_classification.ipynb
│   ├── 03_dl_classification.ipynb
│   ├── 04_graph_preprocessing.ipynb
│   ├── 05_graph_analytics.ipynb
│   └── 06_streaming.ipynb
├── results/
│   └── model_comparison.csv     # Comparative metrics from classical ML runs
└── src/                         # Reusable Python modules
    ├── config.py
    ├── ml_pipeline.py
    ├── utils.py
    ├── utils_dl.py
    └── utils_ml.py
```

---

## Notebooks

- **01_eda.ipynb** – dataset ingestion, quality checks, exploratory analysis, SparkSQL queries,
  and processed output generation for downstream notebooks
- **02_ml_classification.ipynb** – classical Spark ML pipelines, model comparison with shared
  features and splits, confusion matrices, and best-model selection (Linear SVM OvR) for streaming
- **03_dl_classification.ipynb** – distributed DistilBERT fine-tuning via TorchDistributor,
  Spark-native batch inference with `predict_batch_udf`, and per-class evaluation with SparkSQL
- **04_graph_preprocessing.ipynb** – JSONL conversion of the raw DBLP JSON array, schema-on-read,
  venue normalization, FOS feature engineering, data quality reporting, and partitioned Parquet export
- **05_graph_analytics.ipynb** – full-graph and focused subgraph analyses using GraphFrames:
  degree distribution, PageRank, connected components, label propagation, BFS, triangle count,
  shortest paths, and motif finding
- **06_streaming.ipynb** – Kafka ingestion, micro-batch scoring with the selected MLlib pipeline,
  foreachBatch MongoDB upsert, and prediction distribution inspection

---

## Core Python Modules (`src/`)

- **config.py** – centralized paths, Spark defaults, ML hyperparameters, and dataset identifiers
- **utils.py** – Spark session bootstrap and data cleaning / record preparation helpers
- **utils_ml.py** – reusable MLlib training and evaluation utilities; includes LR, SVM OvR, NB,
  Decision Tree, and Random Forest pipeline builders
- **ml_pipeline.py** – standalone configurable Spark ML pipeline assembly and metric evaluation helpers
- **utils_dl.py** – Spark-to-HuggingFace data preparation, DistilBERT training and inference
  helpers for PyTorch

---

## Streaming Infrastructure (`infra/`)

The `infra/` folder contains a Docker Compose stack with four services:

- **tailscale** – secure tailnet exposure of the stack
- **mongodb** – checkpoint and scored-result persistence
- **kafka** – broker for comment stream transport
- **youtube-producer** – polls the YouTube API and publishes normalized comment messages

**Producer behavior:**
- Maintains a per-video checkpoint in MongoDB
- First run backfills all historical comments for the configured video
- Subsequent runs stream only new comments posted after the checkpoint
- Publishes normalized JSON payloads keyed by `comment_id`

**Message schema:**
```json
{
  "comment_id":   "string",
  "video_id":     "string",
  "text":         "string",
  "author":       "string",
  "published_at": "2026-06-02T12:34:56Z",
  "fetched_at":   "2026-06-02T12:35:01Z",
  "source":       "youtube_api"
}
```

**Key streaming details:**
- Kafka topic: `youtube_comments`
- MongoDB write strategy: `foreachBatch` + bulk upsert

---

## Setup

### Prerequisites

- Python 3.11+
- Java runtime compatible with Spark
- Docker + Docker Compose (for the streaming stack)
- Tailscale account and auth key (for infra networking)

### Local Python Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Streaming Infrastructure

```bash
cp infra/.env.example infra/.env
```

Set at minimum:

| Variable | Description |
|---|---|
| `TS_AUTHKEY` | Tailscale auth key |
| `TS_IP_ADDRESS` | Tailnet IP for inter-service routing |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key |
| `YOUTUBE_VIDEO_ID` | Target video to stream comments from |
| `MONGO_USER` | MongoDB username |
| `MONGO_PASS` | MongoDB password |

Then start the stack:

```bash
cd infra && docker compose up -d --build
```

---

## How to Run

Typical execution order:

1. `01_eda.ipynb` – ingest and clean data, generate processed outputs
2. `02_ml_classification.ipynb` – train and compare ML models
3. `03_dl_classification.ipynb` – DistilBERT fine-tuning and evaluation
4. `04_graph_preprocessing.ipynb` – prepare DBLP graph data
5. `05_graph_analytics.ipynb` – run graph algorithms and extract insights
6. Start the infra stack, then run `06_streaming.ipynb` for live scoring

Runtime outputs (not tracked in Git) are written to `data/` and `models/` by the notebooks
and streaming jobs.

---

## Security Notes

- `TS_IP_ADDRESS` is network location metadata, not an application secret by itself.
- Access is gated by tailnet authentication and visibility rules.
- Real credentials (API keys, auth keys, passwords) must remain in env files or a secrets manager
  and must never be committed to the repository.

---

## Known Limitations

- Classical MLlib pipelines can struggle with short or multilingual text — a Portuguese comment
  like *"Lixo"* may be misclassified as Neutral.
- MLlib-first streaming inference improves latency and operational compatibility but trades off
  some accuracy relative to the transformer model.

---

## Future Improvements

- Integrate a stronger multilingual transformer for higher semantic accuracy.
- Add a model/version registry and automated streaming quality monitoring.
- Add reproducible CLI/Makefile tasks for all notebook stages.
- Expand docs with benchmark tables tied to named models in `results/model_comparison.csv`.
