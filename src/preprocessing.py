# src/preprocessing.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

def load_data(file_path: str, spark_session) -> DataFrame:
    """
    Loads raw e-commerce data into a Spark DataFrame.
    """
    df = spark_session.read.option("header", True).csv(file_path)
    return df

def clean_data(df: DataFrame) -> DataFrame:
    """
    Cleans the data by parsing timestamps, removing missing IDs, and filtering events.
    """
    # Convert 'event_time' to timestamp
    df = df.withColumn("event_time", col("event_time").cast("timestamp"))

    # Drop rows missing user_id or session
    df = df.dropna(subset=["user_id", "user_session"])

    # Filter only relevant events
    df = df.filter(df.event_type.isin(["view", "cart", "purchase"]))

    return df



def feature_engineer(df: DataFrame) -> DataFrame:
    """
    Generates session-level features from event-level e-commerce data.

    Args:
        df (DataFrame): Input DataFrame after cleaning.
        
    Returns:
        DataFrame: Session-level aggregated features.
    """
    # Basic aggregates
    session_features = df.groupBy("user_session").agg(
        F.count(F.when(F.col("event_type") == "view", True)).alias("num_views"),
        F.count(F.when(F.col("event_type") == "cart", True)).alias("num_cart_adds"),
        F.count(F.when(F.col("event_type") == "purchase", True)).alias("num_purchases"),
        F.min("event_time").alias("session_start"),
        F.max("event_time").alias("session_end"),
        F.avg("price").alias("avg_price")
    )
    
    # Calculate session duration in seconds
    session_features = session_features.withColumn(
        "session_duration",
        (F.unix_timestamp("session_end") - F.unix_timestamp("session_start"))
    )

    # Fill any nulls (for avg_price or other missing features)
    session_features = session_features.fillna(0)

    return session_features


def assemble_features(df, input_cols, output_col="features"):
    """
    Assembles multiple feature columns into a single 'features' vector column.
    
    Args:
        df (DataFrame): Input DataFrame.
        input_cols (list): List of column names to assemble.
        output_col (str): Name of the output features column.
        
    Returns:
        DataFrame: DataFrame with 'features' column added.
    """
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    df_transformed = assembler.transform(df)
    return df_transformed
