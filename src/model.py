# src/model.py

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from src.save_outputs import save_predictions, save_feature_importances, save_model_metadata
import pandas as pd
import os


def train_random_forest(train_df, features_col="features", label_col="label", numTrees=100, maxDepth=5, maxBins=32):
    """
    Trains a Random Forest model on the provided training DataFrame.
    """
    rf = RandomForestClassifier(featuresCol=features_col, labelCol=label_col,
                                 numTrees=numTrees, maxDepth=maxDepth, maxBins=maxBins)
    model = rf.fit(train_df)
    return model

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

    numTrees, maxDepth, maxBins = best_row['numTrees'], best_row['maxDepth'], best_row['maxBins']
    best_model_name = f"rf_{int(numTrees)}_{int(maxDepth)}_{int(maxBins)}"

    model_dir = os.path.join(base_save_dir, best_model_name)

    paths = {
        "predictions": os.path.join(model_dir, f"{best_model_name}_predictions.csv"),
        "feature_importances": os.path.join(model_dir, f"{best_model_name}_feature_importances.csv"),
        "metadata": os.path.join(model_dir, f"{best_model_name}_metadata.json"),
        "model_name": best_model_name,
        "AUC": best_row["AUC"]
    }

    print(f"🏆 Best Model: {best_model_name} with AUC: {best_row['AUC']:.4f}")
    return paths


def manual_grid_search_rf(train_df, test_df, save_dir="D:/ecomm-bigdata-project/output/rf/"):
    """
    Manually trains Random Forest with different hyperparameters, saves model outputs after each model.
    """
    import os
    import pandas as pd
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    print("🚀 Starting Manual Random Forest grid search with model saves...")

    param_grid = [
        (50, 5, 32),
        (100, 5, 32),
        (200, 5, 32),
        (50, 10, 32),
        (100, 10, 32),
        (200, 10, 32),
        (50, 20, 32),
        (100, 20, 32),
        (200, 20, 32),
        (50, 5, 64),
        (100, 5, 64),
        (200, 5, 64),
        (100, 10, 64),
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

        save_predictions(predictions, os.path.join(model_dir, f"{model_name}_predictions.csv"))
        save_feature_importances(model, os.path.join(model_dir, f"{model_name}_feature_importances.csv"))
        save_model_metadata(model, auc, os.path.join(model_dir, f"{model_name}_metadata.json"))

    print("\n✅ Manual Grid Search Complete! All outputs saved.")

