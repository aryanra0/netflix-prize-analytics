"""
Data Splitter (PySpark Edition).

This script separates the training data from the validation data.
I use the 'probe' set provided by Netflix as my validation set.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from src import config

def create_split():
    # Fire up Spark!
    spark = SparkSession.builder \
        .appName("NetflixPrizeDataSplit") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print("Loading ratings and probe data with PySpark...")
    # Reading from our centralized config paths
    ratings = spark.read.parquet(str(config.RATINGS_FILE))
    probe = spark.read.parquet(str(config.PROBE_FILE))
    
    print("Marking probe set...")
    # Add a flag to the probe data so we can track it after joining
    probe = probe.withColumn("is_probe", lit(True))
    
    # Left join ratings with probe to find which ratings belong to the validation set
    merged = ratings.join(probe, on=['user_id', 'movie_id'], how='left')
    
    # If it didn't match the probe set, it's training data
    merged = merged.fillna(False, subset=['is_probe'])
    
    print("Splitting data...")
    # Filter based on the flag
    train = merged.filter(col("is_probe") == False).drop("is_probe")
    val = merged.filter(col("is_probe") == True).drop("is_probe")
    
    print(f"Saving train to {config.TRAIN_FILE}...")
    train.write.mode("overwrite").parquet(str(config.TRAIN_FILE))
    
    print(f"Saving validation to {config.VAL_FILE}...")
    val.write.mode("overwrite").parquet(str(config.VAL_FILE))
    
    print("Done. Spark is taking a nap now.")
    spark.stop()

if __name__ == "__main__":
    create_split()
