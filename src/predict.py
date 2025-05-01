from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

# Start Spark
spark = SparkSession.builder.appName("StreamlitGBT").getOrCreate()

def predict_purchase(features, modelname, params):
    model_path = f"../output/{modelname}/{modelname}_{params}/model"
    if modelname == "gbt":
        model = GBTClassificationModel.load(model_path)
    else:
        model = RandomForestClassificationModel.load(model_path)
    df = spark.createDataFrame([(Vectors.dense(features), )], ["features"])
    prediction = model.transform(df).collect()[0]
    return float(prediction.probability[1])