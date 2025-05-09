# src/preprocessing.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, split
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window

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
    # Filter nulls and extract main category
    df = df.filter(F.col("category_code").isNotNull())
    df = df.withColumn("category_main", split("category_code", "\\.").getItem(0))

    # --- Basic session-level aggregates ---
    session_features = df.groupBy("user_session").agg(
        F.count(F.when(F.col("event_type") == "view", True)).alias("num_views"),
        F.count(F.when(F.col("event_type") == "cart", True)).alias("num_cart_adds"),
        F.count(F.when(F.col("event_type") == "purchase", True)).alias("num_purchases"),
        F.min("event_time").alias("session_start"),
        F.max("event_time").alias("session_end"),
        F.avg("price").alias("avg_price")
    )

    # --- Session duration in seconds ---
    session_features = session_features.withColumn(
        "session_duration",
        (F.unix_timestamp("session_end") - F.unix_timestamp("session_start"))
    )

    # --- Most frequent main category per session ---
    category_counts = df.groupBy("user_session", "category_main").count()
    window_spec = Window.partitionBy("user_session").orderBy(F.desc("count"))
    ranked = category_counts.withColumn("rank", F.row_number().over(window_spec))
    top_category = ranked.filter(F.col("rank") == 1) \
                         .select("user_session", F.col("category_main").alias("main_category"))

    # --- Category diversity per session ---
    diversity = df.groupBy("user_session").agg(
        F.countDistinct("category_main").alias("unique_categories")
    )

    # --- Join all ---
    session_features = session_features \
        .join(top_category, on="user_session", how="left") \
        .join(diversity, on="user_session", how="left") \
        .fillna({"main_category": "unknown", "unique_categories": 0})

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
