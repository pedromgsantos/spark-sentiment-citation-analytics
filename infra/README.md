# Infra stack

This folder contains the streaming infrastructure used to publish live YouTube comments into Kafka for Spark consumers.

## Services

- `tailscale`: exposes the stack on your Tailscale network.
- `mongodb`: optional persistence service attached to the same Tailscale node.
- `kafka`: single-node Kafka broker in KRaft mode, advertised through the Tailscale IP.
- `youtube-producer`: polls the YouTube Data API and publishes new comments to Kafka.

## MongoDB checkpoint

The YouTube producer uses MongoDB to store one small checkpoint document per video.

- first run: no checkpoint exists, so the producer backfills all historical comments
- later runs: the producer resumes from the saved timestamp

This keeps Kafka as the stream and MongoDB as state only.

## Tailscale access

MongoDB is reachable from any notebook inside your tailnet using the Tailscale IP and port 27017, because the MongoDB container shares the same Tailscale network namespace as the rest of the stack.

Example connection string for a notebook on the tailnet:

```text
mongodb://admin:supersecret@100.122.243.29:27017/?authSource=admin
```

If you use a Spark notebook, you can access MongoDB through a regular Python MongoDB client or a Spark connector configured with that same Tailscale endpoint.

## Message contract

Kafka topic: `youtube_comments`

Each message value is JSON with this shape:

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

The Kafka message key is the YouTube comment ID.

## Startup

1. Update `.env`.
2. Set the machine Tailscale IP in `TS_IP_ADDRESS`.
3. Set a valid `YOUTUBE_API_KEY` and target `YOUTUBE_VIDEO_ID`.
4. Start the stack from this folder with Docker Compose.

## Spark consumer example

Use the Tailscale IP and topic configured in `.env`.

```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType

schema = StructType([
    StructField("comment_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("author", StringType(), True),
    StructField("published_at", StringType(), True),
    StructField("fetched_at", StringType(), True),
    StructField("source", StringType(), True),
])

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "100.x.y.z:9092")
    .option("subscribe", "youtube_comments")
    .option("startingOffsets", "latest")
    .load()
)

comments = raw_stream.select(
    F.col("key").cast("string").alias("comment_key"),
    F.from_json(F.col("value").cast("string"), schema).alias("payload")
).select("comment_key", "payload.*")
```