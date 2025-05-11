import json
import time
from collections import defaultdict
from kafka import KafkaConsumer
import mlflow.pyfunc
from pymongo import MongoClient
import pandas as pd
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import SparseVector

spark = SparkSession.builder.appName("KafkaConsumerVectorAssembly").getOrCreate()

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


category_labels_1 = ['electronics', 'appliances', 'computers', 'apparel', 'furniture', 'auto', 'construction', 'kids', 'accessories', 'sport', 'medicine', 'country_yard', 'stationery']

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

def assemble_features_model1(row_dict, main_category):
    return Vectors.dense(
        [
            float(row_dict["num_views"]),
            float(row_dict["num_cart_adds"]),
            float(row_dict["session_duration"]),
            float(row_dict["avg_price"])
        ] +
        one_hot_category(main_category, category_labels_1) 
    )

def assemble_features_model2(row_dict):
    return Vectors.dense([
        float(row_dict["num_views"]),
        float(row_dict["num_cart_adds"]),
        float(row_dict["avg_price"]),
        float(row_dict["session_duration"]),
    ])

def one_hot_category(cat, all_categories):
    vec = [0.0] * len(all_categories)
    if cat in all_categories:
        vec[all_categories.index(cat)] = 1.0
    return vec


def spark_style_one_hot(index, size):
    """
    Create a SparseVector exactly like Spark's OneHotEncoder output
    :param index: integer index of category
    :param size: total number of categories
    :return: SparseVector with 1.0 at `index`, rest 0
    """
    if index is None or index < 0 or index >= size:
        return SparseVector(size, [], [])
    return SparseVector(size, [index], [1.0])


def prepare_features_for_model1(row_dict, main_category, category_labels, spark):
    """
    Mirrors training feature assembly for model 1.
    
    Parameters:
    - row_dict: dict with raw numeric values
    - main_category: str, extracted from category_code (e.g., "electronics")
    - category_labels: list of categories in training order (indexer.labels)
    - spark: active SparkSession
    
    Returns:
    - pandas DataFrame with a single 'features' column matching model input
    """
    # One-hot encode manually (same as OneHotEncoder with dropLast=True)
    index = category_labels.index(main_category) if main_category in category_labels else -1
    ohe_vector = spark_style_one_hot(index, len(category_labels))

    # Construct raw input row
    raw_row = {
        "num_views": float(row_dict.get("num_views", 0)),
        "num_cart_adds": float(row_dict.get("num_cart_adds", 0)),
        "session_duration": float(row_dict.get("session_duration", 0)),
        "avg_price": float(row_dict.get("avg_price", 0)),
        "unique_categories": float(row_dict.get("unique_categories", 1.0)),
        "main_category_ohe": ohe_vector
    }

    # Create Spark DataFrame
    temp_df = spark.createDataFrame([raw_row])

    # Assemble numeric features
    assembler_numeric = VectorAssembler(
        inputCols=["num_views", "num_cart_adds", "session_duration", "avg_price", "unique_categories"],
        outputCol="numeric_features"
    )
    df_numeric = assembler_numeric.transform(temp_df)

    # Assemble full features
    assembler_all = VectorAssembler(
        inputCols=["numeric_features", "main_category_ohe"],
        outputCol="features"
    )
    final_df = assembler_all.transform(df_numeric)

    # Return as pandas DataFrame with just 'features' column
    return final_df.select("features").toPandas()



print(" Kafka consumer running...")

for msg in consumer:
    event = msg.value
    session_id = event.get("user_session")
    session_buffer[session_id].append(event)

    if len(session_buffer[session_id]) >= 5:  # configurable threshold
        events = session_buffer.pop(session_id)
        features_df = extract_features(events)
        cat_code = str(events[0].get("category_code") or "")
        main_category = cat_code.split(".")[0] if "." in cat_code else "unknown"
        row = features_df.iloc[0]
        # features_df = features_df.copy()
        # features_df["features_model1"] = [assemble_features_model1(row, main_category)]
        # features_df["features_model2"] = [assemble_features_model2(row)]
        input_df_1 = prepare_features_for_model1(row, main_category, category_labels_1, spark)

        try:
            # input_df_1 = features_df[["features_model1"]].rename(columns={"features_model1": "features"})
            # input_df_2 = features_df[["features_model2"]].rename(columns={"features_model2": "features"})

            # print("Model 1 input length:", len(input_df_1.iloc[0]['features']))
            # print("Model 2 input length:", len(input_df_2.iloc[0]['features']))
            # print("Main category for OHE:", main_category)
            # print("Label count for Model 1:", len(category_labels_1))


            purchase_prob = model_1.predict(input_df_1)[0]
            # cat_probs = model_2.predict(input_df_2)[0]
            # top3 = list(np.argsort(cat_probs)[::-1][:3])
            # top3_categories = [label_mapping[int(i)] if i < len(label_mapping) else "unknown" for i in top3]
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            continue

        output = {
            "user_session": session_id,
            "features": features_df.to_dict(orient="records")[0],
            "purchase_prob": purchase_prob,
            # "top3_category_indices": top3,
            # "top3_categories": top3_categories,
            "timestamp": time.time()
        }

        
        collection.insert_one(output)

        print(f" Stored prediction for session: {session_id}")
