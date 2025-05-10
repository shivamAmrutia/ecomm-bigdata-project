import time
import json
import pandas as pd
from kafka import KafkaProducer

# Load your data (change path as needed)
df = pd.read_csv("../data/2019-Nov.csv")
df = df.dropna(subset=["user_id", "product_id", "event_type", "event_time"])

# Convert each row to JSON
def row_to_json(row):
    return json.dumps({
        "user_id": row["user_id"],
        "product_id": row["product_id"],
        "category_id": row.get("category_id"),
        "category_code": row.get("category_code"),
        "brand": row.get("brand"),
        "price": row.get("price"),
        "event_time": row["event_time"],
        "event_type": row["event_type"],
        "user_session": row.get("user_session")
    })

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: v.encode("utf-8")
)

print("🚀 Streaming events to Kafka topic: ecommerce_events")
for idx, row in df.iterrows():
    message = row_to_json(row)
    producer.send("ecommerce_events", value=message)
    print(f"🟢 Sent: {message}")
    time.sleep(0.1)  # Adjust for speed; 0.1s = 10 events/sec

producer.flush()
producer.close()
