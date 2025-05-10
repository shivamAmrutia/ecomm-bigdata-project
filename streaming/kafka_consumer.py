import json
import time
from collections import defaultdict
from kafka import KafkaConsumer
import mlflow.pyfunc
from pymongo import MongoClient
import pandas as pd
import numpy as np

# Kafka setup
consumer = KafkaConsumer(
    'ecommerce_events',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='ecom-predict-group'
)

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["ecommerce"]
collection = db["streamed_predictions"]

# Buffer: user_session -> list of events
session_buffer = defaultdict(list)

# Load pretrained MLflow models
mlflow.set_tracking_uri("file:///D:/ecomm-bigdata-project/mlruns")
model_1 = mlflow.pyfunc.load_model("file:///D:/ecomm-bigdata-project/mlruns/2/14f4fc3af12a42dea4b56286c7caff7f/artifacts/spark-model")
model_2 = mlflow.pyfunc.load_model("file:///D:/ecomm-bigdata-project/mlruns/3/3647e08da90e4334b58a7e1490d9e9f1/artifacts/spark-model")

# Load category label mapping if available
try:
    with open("output/category_labels.txt", "r") as f:
        label_mapping = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    label_mapping = []

def extract_features(events):
    df = pd.DataFrame(events)
    num_views = (df["event_type"] == "view").sum()
    num_cart_adds = (df["event_type"] == "cart").sum()
    avg_price = df["price"].mean()
    timestamps = pd.to_datetime(df["event_time"])
    session_duration = (timestamps.max() - timestamps.min()).total_seconds()
    return pd.DataFrame([{
        "num_views": num_views,
        "num_cart_adds": num_cart_adds,
        "avg_price": avg_price,
        "session_duration": session_duration
    }])

print(" Kafka consumer running...")

for msg in consumer:
    event = msg.value
    print(f"Received event: {event}")
    session_id = event.get("user_session")
    session_buffer[session_id].append(event)

    if len(session_buffer[session_id]) >= 5:  # configurable threshold
        events = session_buffer.pop(session_id)
        features_df = extract_features(events)

        try:
            purchase_prob = model_1.predict(features_df)[0]
            cat_probs = model_2.predict(features_df)[0]
            top3 = list(np.argsort(cat_probs)[::-1][:3])
            top3_categories = [label_mapping[int(i)] if i < len(label_mapping) else "unknown" for i in top3]
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            continue

        output = {
            "user_session": session_id,
            "features": features_df.to_dict(orient="records")[0],
            "purchase_prob": purchase_prob,
            "top3_category_indices": top3,
            "top3_categories": top3_categories,
            "timestamp": time.time()
        }

        
        collection.insert_one(output)

        print(f" Stored prediction for session: {session_id}")
