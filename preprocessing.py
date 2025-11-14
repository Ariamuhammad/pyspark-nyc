from pyspark.sql.functions import (
    col, hour, dayofweek, month, year, radians, sin, cos, atan2, sqrt, lit
)
from pyspark.storagelevel import StorageLevel
from spark_utils import spark_session
from config import RAW_PARQUET, FE_PARQUET, FE_SMALL, FEATURE_COLS, PASSENGER_MIN, PASSENGER_MAX

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Preprocessing")

    df = spark.read.parquet(RAW_PARQUET)
    n_before = df.count()

    # Filter dasar area NYC + nilai yang masuk akal
    filtered = df.where(
        (col("pickup_longitude").between(-79.0, -71.0)) &
        (col("dropoff_longitude").between(-79.0, -71.0)) &
        (col("pickup_latitude").between(38.0, 45.0)) &
        (col("dropoff_latitude").between(38.0, 45.0)) &
        (col("fare_amount") >= 0) &
        (col("passenger_count").between(PASSENGER_MIN, PASSENGER_MAX)) &
        col("pickup_datetime_ts").isNotNull()
    )

    # Haversine distance (km)
    R = lit(6371.0)
    phi1 = radians(col("pickup_latitude"))
    phi2 = radians(col("dropoff_latitude"))
    dphi = radians(col("dropoff_latitude") - col("pickup_latitude"))
    dlmb = radians(col("dropoff_longitude") - col("pickup_longitude"))
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*(sin(dlmb/2)**2)
    c = 2*atan2(sqrt(a), sqrt(1-a))
    dist_km = R*c

    fe = filtered.select(
        col("fare_amount").alias("label"),
        hour(col("pickup_datetime_ts")).alias("pickup_hour"),
        dayofweek(col("pickup_datetime_ts")).alias("pickup_dow"),
        month(col("pickup_datetime_ts")).alias("pickup_month"),
        year(col("pickup_datetime_ts")).alias("pickup_year"),
        col("passenger_count"),
        dist_km.alias("distance_km")
    )

    n_after = fe.count()
    print(f"Baris sebelum filter: {n_before} | sesudah: {n_after} | dibuang: {n_before - n_after}")

    # Simpan full FE dan subset ringan untuk modeling
    fe.write.mode("overwrite").parquet(FE_PARQUET)
    fe.limit(200_000).persist(StorageLevel.MEMORY_AND_DISK).write.mode("overwrite").parquet(FE_SMALL)
    print(f"Tersimpan: {FE_PARQUET} dan subset {FE_SMALL} (200k rows)")

    # Fitur yang terbentuk (untuk tahap modeling berikutnya)
    print("Fitur yang tersedia:", ", ".join(FEATURE_COLS))

    spark.stop()
