# src/model.py

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

    best_model_params = ""
    for col in df.columns:
        best_model_params += str(int(best_row[col])) + "_"
    
    best_model_params = best_model_params[:-1]

    model_dir = os.path.join(base_save_dir, best_model_params)

    paths = {
        "predictions": os.path.join(model_dir, f"predictions.csv"),
        "feature_importances": os.path.join(model_dir, f"feature_importances.csv"),
        "metadata": os.path.join(model_dir, f"metadata.json"),
        "model_name": best_model_params,
        "AUC": best_row["AUC"]
    }

    print(f"🏆 Best Model: {best_model_params} with AUC: {best_row['AUC']:.4f}")

    MLFLow_model_name = str(base_save_dir.split('/')[2])

    # returns like "50_5_32_0,0.9996,rf"
    return str(best_model_params) +"," + str(best_row['AUC']) + "," + MLFLow_model_name

# Random Forest

def manual_grid_search_rf(train_df, test_df, save_dir="../output/rf/"):
    print("🚀 Starting Manual Random Forest grid search...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("rf")

    param_grid = [
        (50, 5, 32), (100, 5, 32), (50, 10, 32), (100, 10, 32), (100, 5, 64), (200, 20, 64)
    ]

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['param1'], df_results['param2'], df_results['param3']))
    else:
        df_results = pd.DataFrame(columns=["param1", "param2", "param3", "AUC"])
        tried_params = set()

    for param1, param2, param3 in param_grid:
        if (param1, param2, param3) in tried_params:
            print(f"⏩ Skipping: param1={param1}, param2={param2}, param3={param3}")
            continue

        print(f"🔵 Training: param1={param1}, param2={param2}, param3={param3}")

        with mlflow.start_run():
            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)
            mlflow.log_param("param3", param3)

            rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                        numTrees=param1, maxDepth=param2, maxBins=param3)
            model = rf.fit(train_df)
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            mlflow.log_metric("auc", float(auc))
            mlflow.spark.log_model(model, artifact_path="spark-model")

            print(f"✔ Logged run with AUC={auc:.4f}")

            new_row = pd.DataFrame([[param1, param2, param3, auc]],
                                   columns=["param1", "param2", "param3", "AUC"])
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_csv(results_csv, index=False)

            model_name = f"rf_{param1}_{param2}_{param3}"
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
            save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
            save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("✅ Random Forest grid search completed.")


# Logistic Regression

def manual_grid_search_lr(train_df, test_df, save_dir="../output/lr/"):
    print("🚀 Starting Manual Logistic Regression grid search...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("lr")

    param_grid = [
        (50, 0.0, 0.0),
        (100, 0.1, 0.0),
        (100, 0.01, 0.5),
        (200, 0.01, 1.0)
    ]

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['param1'], df_results['param2'], df_results['param3']))
    else:
        df_results = pd.DataFrame(columns=["param1", "param2", "param3", "AUC"])
        tried_params = set()

    for param1, param2, param3 in param_grid:
        if (param1, param2, param3) in tried_params:
            print(f"⏩ Skipping: param1={param1}, param2={param2}, param3={param3}")
            continue

        print(f"🔵 Training: param1={param1}, param2={param2}, param3={param3}")

        with mlflow.start_run():
            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)
            mlflow.log_param("param3", param3)

            lr = LogisticRegression(featuresCol="features", labelCol="label",
                                    maxIter=param1, regParam=param2, elasticNetParam=param3)
            model = lr.fit(train_df)
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            mlflow.log_metric("auc", float(auc))
            mlflow.spark.log_model(model, artifact_path="spark-model")

            print(f"✔ Logged run with AUC={auc:.4f}")

            new_row = pd.DataFrame([[param1, param2, param3, auc]],
                                   columns=["param1", "param2", "param3", "AUC"])
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_csv(results_csv, index=False)

            model_name = f"lr_{param1}_{param2}_{param3}"
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
            save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("✅ Logistic Regression grid search completed.")


# Decision Tree

def manual_grid_search_dt(train_df, test_df, save_dir="../output/dt/"):
    print("🚀 Starting Manual Decision Tree grid search...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("dt")

    param_grid = [
        (5, 32), (10, 32), (20, 32), (5, 64), (10, 64)
    ]

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['param1'], df_results['param2']))
    else:
        df_results = pd.DataFrame(columns=["param1", "param2", "AUC"])
        tried_params = set()

    for param1, param2 in param_grid:
        if (param1, param2) in tried_params:
            print(f"⏩ Skipping: param1={param1}, param2={param2}")
            continue

        print(f"🔵 Training: param1={param1}, param2={param2}")

        with mlflow.start_run():
            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)

            dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                        maxDepth=param1, maxBins=param2)
            model = dt.fit(train_df)
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            mlflow.log_metric("auc", float(auc))
            mlflow.spark.log_model(model, artifact_path="spark-model")

            print(f"✔ Logged run with AUC={auc:.4f}")

            new_row = pd.DataFrame([[param1, param2, auc]],
                                   columns=["param1", "param2", "AUC"])
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_csv(results_csv, index=False)

            model_name = f"dt_{param1}_{param2}"
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
            save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
            save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("✅ Decision Tree grid search completed.")


# Naive Bayes

def manual_grid_search_nb(train_df, test_df, save_dir="../output/nb/"):
    print("🚀 Starting Manual Naive Bayes grid search...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nb")

    param_grid = [
        (1.0, "multinomial"),
        (0.5, "multinomial"),
        (2.0, "multinomial")
    ]

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried_params = set(zip(df_results['param1'], df_results['param2']))
    else:
        df_results = pd.DataFrame(columns=["param1", "param2", "AUC"])
        tried_params = set()

    for param1, param2 in param_grid:
        if (param1, param2) in tried_params:
            print(f"⏩ Skipping: param1={param1}, param2={param2}")
            continue

        print(f"🔵 Training: param1={param1}, param2={param2}")

        with mlflow.start_run():
            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)

            nb = NaiveBayes(featuresCol="features", labelCol="label",
                            smoothing=param1, modelType=param2)
            model = nb.fit(train_df)
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            mlflow.log_metric("auc", float(auc))
            mlflow.spark.log_model(model, artifact_path="spark-model")

            print(f"✔ Logged run with AUC={auc:.4f}")

            new_row = pd.DataFrame([[param1, param2, auc]],
                                   columns=["param1", "param2", "AUC"])
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_csv(results_csv, index=False)

            model_name = f"nb_{param1}_{param2}"
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
            save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("✅ Naive Bayes grid search completed.")


# Gradient Boost

def manual_grid_search_gbt(train_df, test_df, save_dir="../output/gbt/"):
    """
    Trains GBTClassifier with multiple hyperparameters, logs models using MLflow, and saves results.
    """

    print("🚀 Starting Manual GBTClassifier grid search...")

    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("gbt")

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
        tried_params = set(zip(df_results['param1'], df_results['param2'], df_results['param3']))
    else:
        df_results = pd.DataFrame(columns=["param1", "param2", "param3", "AUC"])
        tried_params = set()

    for param1, param2, param3 in param_grid:
        if (param1, param2, param3) in tried_params:
            print(f"⏩ Skipping already trained: param1={param1}, param2={param2}, param3={param3}")
            continue

        print(f"🔵 Training: param1={param1}, param2={param2}, param3={param3}")

        with mlflow.start_run():
            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)
            mlflow.log_param("param3", param3)

            gbt = GBTClassifier(featuresCol="features", labelCol="label",
                                maxIter=param1, maxDepth=param2, maxBins=param3)
            model = gbt.fit(train_df)
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            mlflow.log_metric("auc", float(auc))
            mlflow.spark.log_model(model, artifact_path="spark-model")

            print(f"✔ Logged run with AUC={auc:.4f}")

            # Save locally too
            new_row = pd.DataFrame([[param1, param2, param3, auc]], columns=["param1", "param2", "param3", "AUC"])
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_csv(results_csv, index=False)

            model_name = f"gbt_{param1}_{param2}_{param3}"
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            save_predictions(predictions, os.path.join(model_dir, f"predictions.csv"))
            save_feature_importances(model, os.path.join(model_dir, f"feature_importances.csv"))
            save_model_metadata(model, auc, os.path.join(model_dir, f"metadata.json"))

    print("\n✅ Manual Grid Search Complete for GBTClassifier!")


# manual grid search for model 2
def manual_grid_search_category(train_df, test_df, save_dir="../output/category/"):
    print("🚀 Starting Manual Grid Search for Category Prediction...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("category")

    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(save_dir, "grid_search_results.csv")

    # Param grid for each model
    param_configs = [
        {
            "name": "rf",
            "estimator": RandomForestClassifier,
            "params": [(50, 5), (100, 10)],
            "param_names": ["numTrees", "maxDepth"]
        },
        {
            "name": "lr",
            "estimator": LogisticRegression,
            "params": [(50,), (100,)],
            "param_names": ["maxIter"]
        },
        {
            "name": "gbt",
            "estimator": GBTClassifier,
            "params": [(50, 5), (100, 10)],
            "param_names": ["maxIter", "maxDepth"]
        },
        {
            "name": "dt",
            "estimator": DecisionTreeClassifier,
            "params": [(5,), (10,)],
            "param_names": ["maxDepth"]
        }
    ]

    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        tried = set(tuple(row) for row in df_results[["model", "params"]].values)
    else:
        df_results = pd.DataFrame(columns=["model", "params", "accuracy"])
        tried = set()

    for config in param_configs:
        for param_set in config["params"]:
            param_key = (config["name"], str(param_set))
            if param_key in tried:
                print(f"⏩ Skipping {param_key}")
                continue

            print(f"🔵 Training {config['name']} with {param_set}")
            with mlflow.start_run():
                # Set up model with dynamic kwargs
                kwargs = dict(zip(config["param_names"], param_set))
                model = config["estimator"](
                    labelCol="label", featuresCol="features", **kwargs
                )
                mlflow.log_params(kwargs)
                mlflow.log_param("model", config["name"])

                trained_model = model.fit(train_df)
                preds = trained_model.transform(test_df)
                acc = evaluator.evaluate(preds)
                mlflow.log_metric("accuracy", float(acc))
                mlflow.spark.log_model(trained_model, artifact_path="spark-model")

                # Save results
                model_name = f"{config['name']}_" + "_".join(map(str, param_set))
                model_dir = os.path.join(save_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                save_predictions(preds, os.path.join(model_dir, f"predictions.csv"))
                save_model_metadata(trained_model, acc, os.path.join(model_dir, f"metadata.json"))

                if hasattr(trained_model, "featureImportances"):
                    save_feature_importances(trained_model, os.path.join(model_dir, f"feature_importances.csv"))

                new_row = pd.DataFrame([[config["name"], str(param_set), acc]],
                                       columns=["model", "params", "accuracy"])
                df_results = pd.concat([df_results, new_row], ignore_index=True)
                df_results.to_csv(results_csv, index=False)

    print("✅ Category Model Grid Search Completed.")