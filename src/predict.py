from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import mlflow.pyfunc
import pandas as pd
import json

# Start Spark
spark = SparkSession.builder.appName("StreamlitGBT").getOrCreate()

def predict_purchase(features, model, params):
    

    mlflow.set_tracking_uri("http://localhost:5000")

    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(model)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    filtered = runs_df[
    (runs_df['params.maxIter'] == str(params.split('_')[0])) &
    (runs_df['params.maxDepth'] == str(params.split('_')[1])) &
    (runs_df['params.maxBins'] == str(params.split('_')[2]))
    ]

    if not filtered.empty:
        run_id = filtered.iloc[0]['run_id']

    model_uri = f"runs:/{run_id}/spark-model"

    spark_model = mlflow.spark.load_model(model_uri)
    
    df = spark.createDataFrame([(Vectors.dense(features), )], ["features"])
    preds = spark_model.transform(df)
    prob = preds.select("probability").first()[0][1]
    
    return float(prob)
