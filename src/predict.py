from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import mlflow.pyfunc
import pandas as pd
import json

# Start Spark
spark = SparkSession.builder.appName("Streamlit_Predict").getOrCreate()

# Dynamically get category index-to-name mapping from training metadata or source
category_mapping_df = pd.read_csv("./output/category/category_index_mapping.csv")
category_index_to_name = dict(zip(category_mapping_df['index'], category_mapping_df['category']))

def predict_category(num_views, num_cart_adds, session_duration, avg_price):
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name("category")
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])

    if runs_df.empty:
        return "N/A"

    best_run_id = runs_df.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/spark-model"
    model = mlflow.spark.load_model(model_uri)

    # Create input DataFrame
    features = [float(num_views), float(num_cart_adds), 0.0, float(avg_price), float(session_duration), 1.0]
    df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])

    prediction = model.transform(df).select("prediction").first()[0]
    return category_index_to_name.get(int(prediction), "unknown")

def predict_purchase(num_views, num_cart_adds, session_duration, avg_price, selected_category, model, params):
    
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name(model)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    param1, param2, param3 = params.split('_')
    filtered = runs_df[
        (runs_df['params.param1'] == param1) &
        (runs_df['params.param2'] == param2) &
        (runs_df['params.param3'] == param3)
    ]

    run_id = filtered.iloc[0]['run_id']
    model_uri = f"runs:/{run_id}/spark-model"
    spark_model = mlflow.spark.load_model(model_uri)

    # Labels must match the StringIndexer used during training
    category_labels = ['electronics', 'furniture', 'computers', 'kids', 'sport', 'appliances', 'auto', 'unknown']
    
    # One-hot encode selected_category
    ohe_vector = [0.0] * len(category_labels)
    if selected_category in category_labels:
        index = category_labels.index(selected_category)
        ohe_vector[index] = 1.0

    # Full feature vector
    features = [num_views, num_cart_adds, session_duration, avg_price, len(category_labels)]  # use 8 as dummy if unknown
    features.pop()  # remove dummy
    features.extend(ohe_vector)

    df = spark.createDataFrame([(Vectors.dense(features), )], ["features"])
    preds = spark_model.transform(df)
    prob = preds.select("probability").first()[0][1]

    return float(prob)
