from pyspark.sql.functions import col, trim, isnan, avg, stddev, min as smin, max as smax, sum as ssum
from spark_utils import spark_session
from config import RAW_PARQUET

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - EDA")

    df = spark.read.parquet(RAW_PARQUET)
    print(f"Rows: {df.count()}")

    # Hitung NA per kolom (angka dan string), timestamp dihitung isNull saja
    numeric_types = {"double","float","int","bigint","long","decimal","smallint","tinyint"}
    num_cols  = [c for c,t in df.dtypes if t in numeric_types]
    str_cols  = [c for c,t in df.dtypes if t == "string"]
    other     = [c for c,t in df.dtypes if (t not in numeric_types and t != "string")]

    aggs = []
    for c in num_cols:
        aggs.append(ssum((col(c).isNull() | isnan(c)).cast("int")).alias(c))
    for c in str_cols:
        aggs.append(ssum((col(c).isNull() | (trim(col(c))=="")).cast("int")).alias(c))
    for c in other:
        aggs.append(ssum(col(c).isNull().cast("int")).alias(c))

    print("Jumlah NA per kolom:")
    df.agg(*aggs).show(truncate=False)

    if num_cols:
        print("Ringkasan min/max:")
        df.select(*[smin(c).alias(f"{c}_min") for c in num_cols],
                  *[smax(c).alias(f"{c}_max") for c in num_cols]).show(truncate=False)
        print("Rata-rata & stddev:")
        df.select(*[avg(c).alias(f"{c}_avg") for c in num_cols],
                  *[stddev(c).alias(f"{c}_std") for c in num_cols]).show(truncate=False)

    if "passenger_count" in [c for c,_ in df.dtypes]:
        print("Distribusi passenger_count:")
        df.groupBy("passenger_count").count().orderBy("passenger_count").show()

    spark.stop()
