# src/model.py

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

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

def pick_best_model(results_csv_path):
    """
    Picks the best model based on highest AUC score from results CSV.
    """
    df = pd.read_csv(results_csv_path)
    best_model_row = df.loc[df['auc'].idxmax()]
    best_model_name = best_model_row['model_name']
    best_auc = best_model_row['auc']

    print(f"\n🎯 Best model: {best_model_name} with AUC: {best_auc:.4f}")
    return best_model_name, best_auc
