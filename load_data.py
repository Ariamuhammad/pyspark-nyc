import os
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import to_timestamp, col
from spark_utils import spark_session
from config import DATA_CSV, RAW_PARQUET

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Load Data")

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_CSV}")

    schema = StructSchema = StructType([
        StructField("key", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("pickup_datetime", StringType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("passenger_count", IntegerType(), True),
    ])

    df = (spark.read.csv(DATA_CSV, header=True, schema=schema)
                .withColumn("pickup_datetime_ts", to_timestamp(col("pickup_datetime"))))

    print(f"Jumlah baris CSV: {df.count()}")
    df.printSchema()

    # Simpan mentah ke Parquet untuk tahap berikutnya
    df.write.mode("overwrite").parquet(RAW_PARQUET)
    print(f"Tersimpan: {RAW_PARQUET}")

    spark.stop()
