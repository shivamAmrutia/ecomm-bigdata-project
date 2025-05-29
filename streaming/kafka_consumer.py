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

def load_best_model(experiment_name: str,
                    metric_name: str,
                    tracking_uri: str,
                    params:str,
                    artifact_path: str = "spark-model"):
    """
    Fetch the run with the best `metric_name` in the given experiment,
    then load and return its PyFunc model artifact.
    """
    # client = MlflowClient()
    # exp = client.get_experiment_by_name(experiment_name)
    # if exp is None:
    #     raise ValueError(f"Experiment '{experiment_name}' not found")

    # # Search runs, ordering by metric descending (or ascending)
    # order = f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"
    # runs = client.search_runs(
    #     [exp.experiment_id],
    #     order_by=[order],
    #     max_results=1
    # )
    # if not runs:
    #     raise ValueError(f"No runs found in '{experiment_name}'")
    mlflow.set_tracking_uri(tracking_uri)
    
    if params:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs_main_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        param1, param2, param3 = params.split('_')
        runs_df = runs_main_df[
        (runs_main_df['params.param1'] == param1) &
        (runs_main_df['params.param2'] == param2) &
        (runs_main_df['params.param3'] == param3)
        ]
    
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=[f"metrics.{metric_name} DESC"])

    if runs_df.empty:
        return "N/A"

    run_id = runs_df.iloc[0]["run_id"]
    uri = f"runs:/{run_id}/{artifact_path}"
    print(f"> Loading best run {run_id} ( {metric_name} = {metric_name} )")
    return mlflow.pyfunc.load_model(uri)


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
# mlflow.set_tracking_uri("file:///D:/ecomm-bigdata-project/mlruns")
# model_1 = mlflow.pyfunc.load_model("file:///D:/ecomm-bigdata-project/mlruns/2/14f4fc3af12a42dea4b56286c7caff7f/artifacts/spark-model")
# model_2 = mlflow.pyfunc.load_model("file:///D:/ecomm-bigdata-project/mlruns/3/3647e08da90e4334b58a7e1490d9e9f1/artifacts/spark-model")

# use MLflow to pick the best for each experiment:
with open("../output/best_params.json") as f:
    best_model_and_params = json.load(f)
    model_1_name = best_model_and_params.get("model")
    params = best_model_and_params.get("params")

model_1 = load_best_model(
    experiment_name=model_1_name,
    metric_name="auc",
    tracking_uri="http://localhost:5000",
    params=params,
)
model_2 = load_best_model(
    experiment_name="category", 
    metric_name="accuracy",
    tracking_uri="http://localhost:5000",          # or f1 / weightedPrecision, etc.
    params=None
)

# Load category label mapping if available
try:
    with open("output/category_labels.txt", "r") as f:
        label_mapping = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    label_mapping = []


category_labels_1 = ['electronics', 'appliances', 'computers', 'apparel', 'furniture', 'auto', 'construction', 'kids', 'accessories', 'sport', 'medicine', 'country_yard', 'stationery']

def extract_features(events):
    df = pd.DataFrame(events)
    
    def safe_scalar(series, default=0):
        if isinstance(series, pd.Series):
            return float(series.iloc[0]) if not series.empty else default
        return float(series)

    return pd.DataFrame([{
        "num_views": safe_scalar(df["num_views"], 0),
        "num_cart_adds": safe_scalar(df["num_cart_adds"], 0),
        "avg_price": safe_scalar(df["avg_price"], 0),
        "session_duration": safe_scalar(df["session_duration"], 0),
        "num_purchases": safe_scalar(df["num_purchases"], 0),
        "unique_categories": safe_scalar(df["unique_categories"], 1.0),
        "main_category": str(df["main_category"].iloc[0]) if not df["main_category"].empty else "unknown"
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
    return SparseVector(size-1, [index], [1.0])


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
    print(main_category)
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

    print(raw_row)

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

def prepare_features_for_model2(row_dict, spark):
    """
    Mirrors training feature assembly for model 2.
    
    Parameters:
    - row_dict: dict with raw numeric values
    - spark: active SparkSession
    
    Returns:
    - pandas DataFrame with a single 'features' column matching model input
    """
    # 1) Build the raw dict of numeric inputs exactly as in your training pipeline
    raw_row = {
        "num_views": float(row_dict.get("num_views", 0)),
        "num_cart_adds": float(row_dict.get("num_cart_adds", 0)),
        "num_purchases": float(row_dict.get("num_purchases", 1.0)),
        "avg_price": float(row_dict.get("avg_price", 0)),
        "session_duration": float(row_dict.get("session_duration", 0)),
        "unique_categories": float(row_dict.get("unique_categories", 1.0))
    }

    print(raw_row)

    # 2) Convert to a Spark DataFrame
    temp_df = spark.createDataFrame([raw_row])

    # 3) Assemble all numeric columns into the 'features' vector
    assembler = VectorAssembler(
        inputCols=[
            "num_views",
            "num_cart_adds",
            "num_purchases",
            "avg_price",
            "session_duration",
            "unique_categories",
        ],
        outputCol="features"
    )
    final_df = assembler.transform(temp_df)

    # 4) Return just the features column as pandas, for your pyfunc wrapper
    return final_df.select("features").toPandas()




print(" Kafka consumer running...")

for msg in consumer:
    event = msg.value
    session_id = event.get("user_session")
    session_buffer[session_id].append(event)
    if len(session_buffer[session_id]) >= 1:  # configurable threshold
        events = session_buffer.pop(session_id)
        features_df = extract_features(events)
        main_category = str(events[0].get("main_category") or "unknown")
        row = features_df.iloc[0]
        # features_df = features_df.copy()
        # features_df["features_model1"] = [assemble_features_model1(row, main_category)]
        # features_df["features_model2"] = [assemble_features_model2(row)]
        input_df_1 = prepare_features_for_model1(row, main_category, category_labels_1, spark)
        input_df_2 = prepare_features_for_model2(row, spark)

        try:
            # input_df_1 = features_df[["features_model1"]].rename(columns={"features_model1": "features"})
            # input_df_2 = features_df[["features_model2"]].rename(columns={"features_model2": "features"})

            # print("Model 1 input length:", len(input_df_1.iloc[0]['features']))
            # print("Model 2 input length:", len(input_df_2.iloc[0]['features']))
            # print("Main category for OHE:", main_category)
            # print("Label count for Model 1:", len(category_labels_1))


            purchase_prob_np = model_1.predict(input_df_1)[0]
            cat_probs = model_2.predict(input_df_2)[0]
            top3 = np.argsort(cat_probs)[::-1][:3].tolist()
            top3_categories = [label_mapping[int(i)] if i < len(label_mapping) else "unknown" for i in top3]

            #turn logging values into native data types
            purchase_prob = float(purchase_prob_np)
            

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
