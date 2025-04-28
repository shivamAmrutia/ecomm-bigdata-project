def save_predictions(predictions_df, output_path):
    """
    Saves prediction results to CSV.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.select("user_session", "prediction", "probability") \
                  .toPandas() \
                  .to_csv(output_path, index=False)
    print(f"✅ Predictions saved: {output_path}")

def save_feature_importances(model, output_path):
    """
    Saves feature importances from a trained Random Forest model.
    """
    import pandas as pd
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fi = model.featureImportances.toArray()
    fi_df = pd.DataFrame(fi, columns=["importance"])
    fi_df.to_csv(output_path, index=False)
    print(f"✅ Feature importances saved: {output_path}")

def save_model_metadata(model, auc_score, output_path):
    """
    Saves model metadata and AUC to a JSON file.
    """
    import json
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {
        "numTrees": model.getNumTrees,
        "maxDepth": model.getOrDefault("maxDepth"),
        "maxBins": model.getOrDefault("maxBins"),
        "AUC": auc_score
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Model metadata saved: {output_path}")
