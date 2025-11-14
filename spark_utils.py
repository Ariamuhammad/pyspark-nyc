import os
from pyspark.sql import SparkSession

def spark_session(app_name="NYC Taxi - Step", driver_mem="6g", shuffle_parts="60"):
    os.makedirs("artifacts", exist_ok=True)
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
