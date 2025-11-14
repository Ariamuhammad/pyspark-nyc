import os
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql import functions as F
from spark_utils import spark_session
from config import DATA_CSV, RAW_PARQUET, PASSENGER_MIN, PASSENGER_MAX


def report_data_quality(df):
    
    geo_valid = (
        F.col("pickup_longitude").between(-79.0, -71.0) &
        F.col("dropoff_longitude").between(-79.0, -71.0) &
        F.col("pickup_latitude").between(38.0, 45.0) &
        F.col("dropoff_latitude").between(38.0, 45.0)
    )
    aggs = [
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("fare_amount") < 0, 1).otherwise(0)).alias("fare_negative_rows"),
        F.sum(F.when(
            (F.col("passenger_count") < PASSENGER_MIN) | (F.col("passenger_count") > PASSENGER_MAX)
            | F.col("passenger_count").isNull(),
            1
        ).otherwise(0)).alias("passenger_count_out_of_range"),
        F.sum(F.when(~geo_valid, 1).otherwise(0)).alias("geo_out_of_bounds_rows"),
    ]
    required_cols = [
        "fare_amount", "pickup_datetime_ts", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "passenger_count"
    ]
    for col_name in required_cols:
        aggs.append(
            F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(f"{col_name}_nulls")
        )

    stats = df.agg(*aggs).collect()[0].asDict()
    total_rows = stats.pop("total_rows", 0)
    print("Ringkasan kualitas data awal:")
    for key, value in stats.items():
        pct = (value / total_rows * 100) if total_rows else 0
        print(f"  {key:30s}: {value:6d} ({pct:6.2f}% dari data)")
    return total_rows

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Load Data")

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_CSV}")

    schema = StructType([
        StructField("key", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("pickup_datetime", StringType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("passenger_count", IntegerType(), True),
    ])

    df = (
        spark.read.csv(DATA_CSV, header=True, schema=schema)
        .withColumn("pickup_datetime_ts", F.to_timestamp(F.col("pickup_datetime")))
    )

    total_rows_initial = df.count()
    if total_rows_initial == 0:
        raise ValueError("Dataset kosong. Periksa kembali file train.csv yang digunakan.")
    print(f"Jumlah baris awal (raw CSV): {total_rows_initial}")

    distinct_keys = df.select("key").distinct().count()
    duplicate_rows = total_rows_initial - distinct_keys
    if duplicate_rows > 0:
        print(f"Deteksi duplikat: {duplicate_rows} baris dengan key identik akan dihapus.")
        df = df.dropDuplicates(["key"])
    else:
        print("Deteksi duplikat: tidak ditemukan baris dengan key identik.")

    total_rows = report_data_quality(df)
    if total_rows == 0:
        raise ValueError("Semua baris terfilter setelah validasi kualitas. Periksa aturan yang diterapkan.")
    print(f"Jumlah baris setelah pembersihan: {total_rows}")
    df.printSchema()

    # Simpan mentah ke Parquet untuk tahap berikutnya
    df.write.mode("overwrite").parquet(RAW_PARQUET)
    print(f"Tersimpan: {RAW_PARQUET}")

    spark.stop()
