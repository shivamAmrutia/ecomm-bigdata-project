# src/model.py

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from src.save_outputs import save_predictions, save_feature_importances, save_model_metadata
import pandas as pd
import os

import mlflow
import mlflow.spark


def evaluate_model(model, test_df, label_col="label"):
    """
    Evaluates a trained model using AUC metric on test dataset.
    """
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    predictions = model.transform(test_df)
    auc = evaluator.evaluate(predictions)
    return auc

def pick_best_model_from_grid(results_csv_path, base_save_dir):
    """
    Picks best model based on highest AUC from grid search results and returns paths to best model outputs.
    
    Args:
        results_csv_path (str): Path to saved grid_search_results.csv.
        base_save_dir (str): Base directory where models are saved.
    
    Returns:
        dict: Paths to best model's predictions, features, metadata.
    """
    df = pd.read_csv(results_csv_path)
    best_row = df.loc[df['AUC'].idxmax()]

    best_model_name = ""
    for col in df.columns:
        best_model_name += str(int(best_row[col])) + "_"
    
    best_model_name = best_model_name[:-1]

    model_dir = os.path.join(base_save_dir, best_model_name)

    paths = {
        "predictions": os.path.join(model_dir, f"predictions.csv"),
        "feature_importances": os.path.join(model_dir, f"feature_importances.csv"),
        "metadata": os.path.join(model_dir, f"metadata.json"),
        "model_name": best_model_name,
        "AUC": best_row["AUC"]
    }

    print(f"🏆 Best Model: {best_model_name} with AUC: {best_row['AUC']:.4f}")
    # returns like "50_5_32_0,0.9996,rf"
    return str(best_model_name) +"," + str(best_row['AUC']) + "," + str(base_save_dir.split('/')[2])

# Random Forest

def manual_grid_search_rf(train_df, test_df, save_dir="../output/rf/"):
    """
    Manually trains Random Forest with different hyperparameters, saves model outputs after each model.
    """

    print("🚀 Starting Manual Random Forest grid search with model saves...")

    param_grid = [
        (50, 5, 32),
        (100, 5, 32),
        # (200, 5, 32),
        (50, 10, 32),
        (100, 10, 32),
        # (200, 10, 32),
        # (50, 20, 32),
        # (100, 20, 32),
        # (200, 20, 32),
        # (50, 5, 64),
        (100, 5, 64),
        # (200, 5, 64),
        # (100, 10, 64),
        (200, 20, 64)
    ]

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    # Load existing results if any
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['numTrees'], df_results['maxDepth'], df_results['maxBins']))
    else:
        df_results = pd.DataFrame(columns=["numTrees", "maxDepth", "maxBins", "AUC"])
        tried_params = set()

    for numTrees, maxDepth, maxBins in param_grid:
        if (numTrees, maxDepth, maxBins) in tried_params:
            print(f"⏩ Skipping already trained: numTrees={numTrees}, maxDepth={maxDepth}, maxBins={maxBins}")
            continue

        print(f"🔵 Training: numTrees={numTrees}, maxDepth={maxDepth}, maxBins={maxBins}")

        rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                     numTrees=numTrees, maxDepth=maxDepth, maxBins=maxBins)
        model = rf.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"🔹 AUC: {auc:.4f}")

        # Save results
        new_row = pd.DataFrame([[numTrees, maxDepth, maxBins, auc]], columns=["numTrees", "maxDepth", "maxBins", "AUC"])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(results_csv, index=False)

        # Auto-save model outputs
        model_name = f"rf_{numTrees}_{maxDepth}_{maxBins}"
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
        save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete! All outputs saved.")

# Linear Regression

def manual_grid_search_lr(train_df, test_df, save_dir="../output/lr/"):
    """
    Manually trains Logistic Regression with different hyperparameters, saves model outputs after each model.
    """

    print("🚀 Starting Manual Logistic Regression grid search...")

    param_grid = [
        (50, 0.0, 0.0),
        (100, 0.1, 0.0),
        (100, 0.01, 0.5),
        (200, 0.01, 1.0)
    ]  # (maxIter, regParam, elasticNetParam)

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['maxIter'], df_results['regParam'], df_results['elasticNetParam']))
    else:
        df_results = pd.DataFrame(columns=["maxIter", "regParam", "elasticNetParam", "AUC"])
        tried_params = set()

    for maxIter, regParam, elasticNetParam in param_grid:
        if (maxIter, regParam, elasticNetParam) in tried_params:
            print(f"⏩ Skipping already trained: maxIter={maxIter}, regParam={regParam}, elasticNetParam={elasticNetParam}")
            continue

        print(f"🔵 Training: maxIter={maxIter}, regParam={regParam}, elasticNetParam={elasticNetParam}")

        lr = LogisticRegression(featuresCol="features", labelCol="label",
                                 maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
        model = lr.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"🔹 AUC: {auc:.4f}")

        new_row = pd.DataFrame([[maxIter, regParam, elasticNetParam, auc]], columns=["maxIter", "regParam", "elasticNetParam", "AUC"])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(results_csv, index=False)

        model_name = f"lr_{maxIter}_{regParam}_{elasticNetParam}"
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete for Logistic Regression!")


# Decision Tree

def manual_grid_search_dt(train_df, test_df, save_dir="../output/dt/"):
    """
    Manually trains Decision Tree with different hyperparameters, saves model outputs after each model.
    """

    print("🚀 Starting Manual Decision Tree grid search...")

    param_grid = [
        (5, 32),
        (10, 32),
        (20, 32),
        (5, 64),
        (10, 64)
    ]  # (maxDepth, maxBins)

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['maxDepth'], df_results['maxBins']))
    else:
        df_results = pd.DataFrame(columns=["maxDepth", "maxBins", "AUC"])
        tried_params = set()

    for maxDepth, maxBins in param_grid:
        if (maxDepth, maxBins) in tried_params:
            print(f"⏩ Skipping already trained: maxDepth={maxDepth}, maxBins={maxBins}")
            continue

        print(f"🔵 Training: maxDepth={maxDepth}, maxBins={maxBins}")

        dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                    maxDepth=maxDepth, maxBins=maxBins)
        model = dt.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"🔹 AUC: {auc:.4f}")

        new_row = pd.DataFrame([[maxDepth, maxBins, auc]], columns=["maxDepth", "maxBins", "AUC"])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(results_csv, index=False)

        model_name = f"dt_{maxDepth}_{maxBins}"
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
        save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete for Decision Tree!")


# Naive Bayes

def manual_grid_search_nb(train_df, test_df, save_dir="../output/nb/"):
    """
    Manually trains Naive Bayes with different hyperparameters, saves model outputs after each model.
    """

    print("🚀 Starting Manual Naive Bayes grid search...")

    param_grid = [
        (1.0, "multinomial"),
        (0.5, "multinomial"),
        (2.0, "multinomial")
    ]  # (smoothing, modelType)

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['smoothing'], df_results['modelType']))
    else:
        df_results = pd.DataFrame(columns=["smoothing", "modelType", "AUC"])
        tried_params = set()

    for smoothing, modelType in param_grid:
        if (smoothing, modelType) in tried_params:
            print(f"⏩ Skipping already trained: smoothing={smoothing}, modelType={modelType}")
            continue

        print(f"🔵 Training: smoothing={smoothing}, modelType={modelType}")

        nb = NaiveBayes(featuresCol="features", labelCol="label",
                        smoothing=smoothing, modelType=modelType)
        model = nb.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"🔹 AUC: {auc:.4f}")

        new_row = pd.DataFrame([[smoothing, modelType, auc]], columns=["smoothing", "modelType", "AUC"])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(results_csv, index=False)

        model_name = f"nb_{smoothing}_{modelType}"
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete for Naive Bayes!")


# Gradient Boost

def manual_grid_search_gbt(train_df, test_df, save_dir="../output/gbt/"):
    """
    Manually trains GBTClassifier with different hyperparameters, saves model outputs after each model.
    """

    print("🚀 Starting Manual GBTClassifier grid search...")

    param_grid = [
        (50, 5, 32),
        (100, 5, 32),
        (50, 10, 64),
        (100, 10, 64)
    ]  # (maxIter, maxDepth, maxBins)

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['maxIter'], df_results['maxDepth'], df_results['maxBins']))
    else:
        df_results = pd.DataFrame(columns=["maxIter", "maxDepth", "maxBins", "AUC"])
        tried_params = set()

    for maxIter, maxDepth, maxBins in param_grid:
        if (maxIter, maxDepth, maxBins) in tried_params:
            print(f"⏩ Skipping already trained: maxIter={maxIter}, maxDepth={maxDepth}, maxBins={maxBins}")
            continue

        print(f"🔵 Training: maxIter={maxIter}, maxDepth={maxDepth}, maxBins={maxBins}")

        gbt = GBTClassifier(featuresCol="features", labelCol="label",
                            maxIter=maxIter, maxDepth=maxDepth, maxBins=maxBins)
        model = gbt.fit(train_df)
        predictions = model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"🔹 AUC: {auc:.4f}")

        new_row = pd.DataFrame([[maxIter, maxDepth, maxBins, auc]], columns=["maxIter", "maxDepth", "maxBins", "AUC"])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(results_csv, index=False)

        model_name = f"gbt_{maxIter}_{maxDepth}_{maxBins}"
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
        save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete for GBTClassifier!")

def train_and_log_mlflow(train_df, test_df,
                         maxIter: int,
                         regParam: float = 0.0,
                         elasticNetParam: float = 0.0):
    # 1) Point at our server & experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test_lr_experiment")

    with mlflow.start_run():
        # 2) Log your params
        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)
        mlflow.log_param("elasticNetParam", elasticNetParam)

        # 3) Train
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=maxIter,
            regParam=regParam,
            elasticNetParam=elasticNetParam
        )
        model = lr.fit(train_df)

        # 4) Predict & evaluate
        preds = model.transform(test_df)
        evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                  rawPredictionCol="probability",
                                                  metricName="areaUnderROC")
        auc = evaluator.evaluate(preds)
        mlflow.log_metric("auc", float(auc))

        # 5) Log the Spark model as an artifact
        mlflow.spark.log_model(model, artifact_path="spark-model")

        print(f"✔ Logged run with AUC={auc:.4f}")