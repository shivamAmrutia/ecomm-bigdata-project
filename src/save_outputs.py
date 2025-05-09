from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
import json
import os
import pandas as pd

def save_predictions(predictions_df, output_path):
    """
    Saves prediction results to CSV with extracted probability[1].
    """
    # UDF to extract probability of class 1
    extract_prob = udf(lambda prob: float(prob[1]), DoubleType())

    cleaned = predictions_df.withColumn("probability_class1", extract_prob(col("probability")))

    selected_cols = ["user_session", "num_views", "num_cart_adds", "label",
                     "prediction", "probability_class1", "session_duration", "avg_price", 'main_category_ohe', 'unique_categories']

    # Save as CSV
    cleaned.select(*selected_cols) \
           .toPandas() \
           .to_csv(output_path, index=False)

    print(f"✅ Predictions saved with class 1 probabilities: {output_path}")


def save_feature_importances(model, output_path):
    """
    Saves feature importances from a trained Random Forest model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fi = model.featureImportances.toArray()
    fi_df = pd.DataFrame(fi, columns=["importance"])
    fi_df.to_csv(output_path, index=False)
    print(f"✅ Feature importances saved: {output_path}")

def save_model_metadata(model, auc_score, output_path):
    """
    Saves model metadata and AUC to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    param_map = model.extractParamMap()
    param_dict = {str(k.name): v for k, v in param_map.items()}

    metadata = {
        "model_params": param_dict,
        "AUC": auc_score
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Model metadata saved: {output_path}")
