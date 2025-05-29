import time
import json
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: v.encode("utf-8")
)

print("🚀 Starting Kafka Producer")

csv_path = "../data/streamingData/streamingData.csv"
chunk_size = 1000

def row_to_json(row):
    return json.dumps({
        "user_session": row.get("user_session"),
        "num_views": row.get("num_views"),
        "num_cart_adds": row.get("num_cart_adds"),
        "num_purchases": row.get("num_purchases"),
        "session_start": row.get("session_start"),
        "session_end": row.get("session_end"),
        "avg_price": row.get("avg_price"),
        "session_duration": row.get("session_duration"),
        "main_category": row.get("main_category"),
        "unique_categories": row.get("unique_categories"),
        "label_category": row.get("label_category")
    })

chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)

for chunk_num, chunk in enumerate(chunk_iter):
    chunk = chunk.dropna(subset=["user_session", "main_category"])
    print(f"📦 Processing chunk {chunk_num} with {len(chunk)} rows")

    for idx, row in chunk.iterrows():
        msg = row_to_json(row)
        producer.send("ecommerce_events", value=msg)
        print(f"🟢 Sent: {msg}")
        time.sleep(1)  # Adjust rate as needed

producer.flush()
producer.close()
print("✅ Finished streaming all chunks")
