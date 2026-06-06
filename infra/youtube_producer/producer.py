import os
import time
import json
import datetime
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError

# Configure standard logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID", "dQw4w9WgXcQ")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "youtube_stream")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "producer_state")
MONGO_STARTUP_TIMEOUT_SECONDS = int(os.getenv("MONGO_STARTUP_TIMEOUT_SECONDS", "120"))
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "youtube_comments")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
KAFKA_STARTUP_TIMEOUT_SECONDS = int(os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "120"))


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def to_rfc3339(dt_value):
    return dt_value.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_rfc3339(timestamp):
    return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


if not API_KEY:
    raise RuntimeError("Missing required environment variable: YOUTUBE_API_KEY")
if not MONGO_USER or not MONGO_PASS:
    raise RuntimeError("Missing required environment variables: MONGO_USER and MONGO_PASS")

logger.info("Initializing YouTube API client...")
youtube = build("youtube", "v3", developerKey=API_KEY)

def build_mongo_uri():
    return f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"


def wait_for_mongo():
    deadline = time.time() + MONGO_STARTUP_TIMEOUT_SECONDS
    last_error = None

    while time.time() < deadline:
        try:
            client = MongoClient(build_mongo_uri(), serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            logger.info("Connected to MongoDB at %s:%s", MONGO_HOST, MONGO_PORT)
            return client
        except (ServerSelectionTimeoutError, PyMongoError) as exc:
            last_error = exc
            logger.info("MongoDB not ready yet, retrying in 5 seconds...")
            time.sleep(5)

    raise RuntimeError(f"Failed to connect to MongoDB within timeout: {last_error}")


mongo_client = wait_for_mongo()
state_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def load_checkpoint():
    record = state_collection.find_one({"video_id": VIDEO_ID})
    if not record:
        return None

    last_fetched_value = record.get("last_fetched_time")
    if not last_fetched_value:
        return None

    return parse_rfc3339(last_fetched_value)


def save_checkpoint(timestamp, first_run_completed):
    state_collection.update_one(
        {"video_id": VIDEO_ID},
        {
            "$set": {
                "video_id": VIDEO_ID,
                "last_fetched_time": to_rfc3339(timestamp),
                "first_run_completed": first_run_completed,
                "updated_at": to_rfc3339(utc_now()),
            }
        },
        upsert=True,
    )


last_fetched_time = load_checkpoint()
first_run = last_fetched_time is None
if first_run:
    logger.info("No MongoDB checkpoint found for video %s. First run will backfill history.", VIDEO_ID)
else:
    logger.info("Loaded MongoDB checkpoint: %s", to_rfc3339(last_fetched_time))


def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        retries=10,
        linger_ms=250,
    )


def ensure_topic_exists():
    admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    try:
        admin_client.create_topics(
            new_topics=[NewTopic(name=TOPIC_NAME, num_partitions=1, replication_factor=1)],
            validate_only=False,
        )
        logger.info("Created Kafka topic '%s'", TOPIC_NAME)
    except TopicAlreadyExistsError:
        logger.info("Kafka topic '%s' already exists", TOPIC_NAME)
    finally:
        admin_client.close()


def wait_for_kafka():
    deadline = time.time() + KAFKA_STARTUP_TIMEOUT_SECONDS
    last_error = None

    while time.time() < deadline:
        try:
            ensure_topic_exists()
            kafka_producer = create_kafka_producer()
            logger.info("Connected to Kafka broker at %s", KAFKA_BROKER)
            return kafka_producer
        except NoBrokersAvailable as exc:
            last_error = exc
            logger.info("Kafka not ready yet, retrying in 5 seconds...")
            time.sleep(5)
        except Exception as exc:
            last_error = exc
            logger.warning("Kafka connection attempt failed: %s", exc)
            time.sleep(5)

    raise RuntimeError(f"Failed to connect to Kafka within timeout: {last_error}")


producer = wait_for_kafka()
logger.info("Tracking video ID: %s", VIDEO_ID)
if last_fetched_time is not None:
    logger.info("Initial checkpoint: %s", to_rfc3339(last_fetched_time))

def fetch_and_stream_new_comments():
    global last_fetched_time, first_run

    try:
        page_token = None
        pending_comments = []
        newest_timestamp_this_batch = None
        reached_checkpoint = False

        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=VIDEO_ID,
                maxResults=100,
                order="time",
                pageToken=page_token,
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                published_at = parse_rfc3339(comment["publishedAt"])

                if last_fetched_time is not None and published_at <= last_fetched_time:
                    reached_checkpoint = True
                    break

                if newest_timestamp_this_batch is None or published_at > newest_timestamp_this_batch:
                    newest_timestamp_this_batch = published_at
                pending_comments.append((published_at, item))

            if reached_checkpoint or not response.get("nextPageToken"):
                break

            page_token = response.get("nextPageToken")

        comments_sent = 0
        for published_at, item in sorted(pending_comments, key=lambda row: row[0]):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            payload = {
                "comment_id": item["id"],
                "video_id": VIDEO_ID,
                "text": comment.get("textOriginal", ""),
                "author": comment.get("authorDisplayName", "unknown"),
                "published_at": to_rfc3339(published_at),
                "fetched_at": to_rfc3339(utc_now()),
                "source": "youtube_api",
            }

            producer.send(TOPIC_NAME, key=item["id"], value=payload)
            comments_sent += 1

        producer.flush()

        if comments_sent:
            last_fetched_time = newest_timestamp_this_batch or utc_now()
            save_checkpoint(last_fetched_time, first_run_completed=True)
            first_run = False
            logger.info(
                "Streamed %s comments to topic '%s'. Checkpoint updated to %s",
                comments_sent,
                TOPIC_NAME,
                to_rfc3339(last_fetched_time),
            )
        else:
            if first_run and last_fetched_time is None:
                last_fetched_time = utc_now()
                save_checkpoint(last_fetched_time, first_run_completed=True)
                first_run = False
                logger.info("No comments found during backfill. Stored initial checkpoint at %s", to_rfc3339(last_fetched_time))
                return

            logger.info("No new comments found for video %s", VIDEO_ID)

    except HttpError as e:
        logger.error(f"YouTube API Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during fetch: {e}")

if __name__ == "__main__":
    while True:
        fetch_and_stream_new_comments()
        time.sleep(POLL_INTERVAL_SECONDS)