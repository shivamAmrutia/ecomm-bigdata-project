import time
import json
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: v.encode("utf-8")
)

print("🚀 Starting Kafka Producer")

csv_path = "../data/2019-Nov.csv"
chunk_size = 1000

def row_to_json(row):
    return json.dumps({
        "user_id": row.get("user_id"),
        "product_id": row.get("product_id"),
        "category_id": row.get("category_id"),
        "category_code": row.get("category_code"),
        "brand": row.get("brand"),
        "price": row.get("price"),
        "event_time": row.get("event_time"),
        "event_type": row.get("event_type"),
        "user_session": row.get("user_session")
    })

chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)

for chunk_num, chunk in enumerate(chunk_iter):
    chunk = chunk.dropna(subset=["user_id", "product_id", "event_type", "event_time", "user_session"])
    print(f"📦 Processing chunk {chunk_num} with {len(chunk)} rows")

    for idx, row in chunk.iterrows():
        msg = row_to_json(row)
        producer.send("ecommerce_events", value=msg)
        print(f"🟢 Sent: {msg}")
        time.sleep(0.05)  # Adjust rate as needed

producer.flush()
producer.close()
print("✅ Finished streaming all chunks")
