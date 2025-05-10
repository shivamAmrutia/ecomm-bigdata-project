from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, count, avg, min, max, expr, udf, current_timestamp
from pyspark.sql.types import StructType, StringType, DoubleType, ArrayType
from pyspark.ml.feature import VectorAssembler
import mlflow.spark
import numpy as np

# Step 1: Initialize Spark
spark = SparkSession.builder \
    .appName("KafkaStreamConsumer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Step 2: Define schema for incoming JSON data
schema = StructType() \
    .add("user_id", StringType()) \
    .add("product_id", StringType()) \
    .add("category_id", StringType()) \
    .add("category_code", StringType()) \
    .add("brand", StringType()) \
    .add("price", DoubleType()) \
    .add("event_time", StringType()) \
    .add("event_type", StringType()) \
    .add("user_session", StringType())

# Step 3: Read from Kafka
raw_stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ecommerce_events") \
    .option("startingOffsets", "latest") \
    .load()

# Step 4: Decode Kafka messages (value = JSON string)
json_df = raw_stream_df.selectExpr("CAST(value AS STRING)")

# Step 5: Parse JSON into structured columns
parsed_df = json_df.select(from_json(col("value"), schema).alias("data")).select("data.*")


parsed_df = parsed_df.withColumn("event_time", to_timestamp("event_time"))

session_features = parsed_df.groupBy("user_session").agg(
    count(expr("event_type = 'view'")).alias("num_views"),
    count(expr("event_type = 'cart'")).alias("num_cart_adds"),
    avg("price").alias("avg_price"),
    min("event_time").alias("session_start"),
    max("event_time").alias("session_end")
).withColumn("session_duration", expr("unix_timestamp(session_end) - unix_timestamp(session_start)"))


#assemble features for model 1
feature_cols = ["num_views", "num_cart_adds", "avg_price", "session_duration"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_df = assembler.transform(session_features)


#load pretrained models
model_1_uri = "runs:/<your_run_id_1>/spark-model" 
model_2_uri = "runs:/<your_run_id_2>/spark-model" 

model_1 = mlflow.spark.load_model(model_1_uri)
model_2 = mlflow.spark.load_model(model_2_uri)

preds_1 = model_1.transform(assembled_df)
preds_2 = model_2.transform(assembled_df)


# Probability of purchase
extract_prob = udf(lambda v: float(v[1]), DoubleType())
preds_1 = preds_1.withColumn("purchase_prob", extract_prob("probability"))

# Top-3 category predictions
get_top3 = udf(lambda v: np.argsort(v)[::-1][:3].tolist(), ArrayType(DoubleType()))
preds_2 = preds_2.withColumn("top3_category_indices", get_top3("probability"))

# 🔁 Placeholder for category label mapping
# label_mapping should be a list like: ['electronics', 'furniture', 'computers', ...]
# It should be saved during Model 2 training using: indexer_model.labels
# Load and broadcast when available

# Example:
# with open("output/category_labels.txt", "r") as f:
#     label_mapping = [line.strip() for line in f.readlines()]
# broadcast_map = spark.sparkContext.broadcast(label_mapping)

# Then decode:
# decode_udf = udf(lambda indices: [broadcast_map.value[int(i)] for i in indices], ArrayType(StringType()))
# preds_2 = preds_2.withColumn("top3_categories", decode_udf("top3_category_indices"))


# Step 6: Preview parsed stream
final_output = preds_1.join(preds_2, on="user_session")

# Select columns you want to store
mongo_df = final_output.select("user_session", "purchase_prob", "top3_category_indices")

mongo_df = mongo_df.withColumn("timestamp", current_timestamp())

# Write to MongoDB
mongo_query = mongo_df.writeStream \
    .format("mongodb") \
    .option("spark.mongodb.connection.uri", "mongodb://localhost:27017") \
    .option("spark.mongodb.database", "ecommerce") \
    .option("spark.mongodb.collection", "streamed_predictions") \
    .outputMode("append") \
    .start()

mongo_query.awaitTermination()

