"""
Feature Engineering (PySpark Edition).

I compute statistical features for users and movies to enhance model performance.
This script calculates metrics such as User Mean Rating and Movie Rating Count.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, count, stddev
from src import config

def compute_features():
    spark = SparkSession.builder \
        .appName("NetflixPrizeFeaturize") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print(f"Loading ratings from {config.TRAIN_FILE}...")
    df = spark.read.parquet(str(config.TRAIN_FILE))
    
    print("Computing User Features (Mean, Count, StdDev)...")
    # Group by user and crunch numbers
    user_stats = df.groupBy('user_id').agg(
        mean('rating').alias('user_mean'),
        count('rating').alias('user_count'),
        stddev('rating').alias('user_std')
    ).fillna(0) # Fill NaNs (e.g. if stddev is undefined for single rating)
    
    print("Computing Item Features...")
    # Group by movie and crunch numbers
    item_stats = df.groupBy('movie_id').agg(
        mean('rating').alias('movie_mean'),
        count('rating').alias('movie_count'),
        stddev('rating').alias('movie_std')
    ).fillna(0)
    
    print("Saving features to disk...")
    user_stats.write.mode("overwrite").parquet(str(config.USER_FEATURES_FILE))
    item_stats.write.mode("overwrite").parquet(str(config.ITEM_FEATURES_FILE))
    
    print("Done. Features are ready for the XGBoost beast.")
    spark.stop()

if __name__ == "__main__":
    compute_features()
