# main.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import (
    to_timestamp, col, trim, isnan, count as scount, avg, stddev,
    min as smin, max as smax, sum as ssum,
    radians, sin, cos, atan2, sqrt, lit, hour, dayofweek, month, year
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ========= 0) SparkSession =========
# Jika JAVA_HOME sudah benar ke Java 11, ini langsung jalan.
spark = SparkSession.builder \
    .appName("NYC Taxi Fare - PySpark (Local VS Code)") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark version:", spark.version)

# ========= 1) Load data =========
DATA_FP = os.path.join("data", "train.csv")
if not os.path.exists(DATA_FP):
    raise FileNotFoundError(f"train.csv tidak ditemukan di {DATA_FP}. Pastikan file ada di folder data/")

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

train_df = spark.read.csv(DATA_FP, header=True, schema=schema)
train_df = train_df.withColumn("pickup_datetime_ts", to_timestamp(col("pickup_datetime")))
train_df.cache()
print("Jumlah baris train:", train_df.count())
train_df.printSchema()

# ========= 2) EDA ringkas =========
numeric_types = {"double","float","int","bigint","long","decimal","smallint","tinyint"}
num_cols = [c for c,t in train_df.dtypes if t in numeric_types]
str_cols = [c for c,t in train_df.dtypes if t == "string"]
other_cols = [c for c,t in train_df.dtypes if (t not in numeric_types and t != "string")]  # contoh: timestamp

aggs = []
for c in num_cols:
    aggs.append(ssum((col(c).isNull() | isnan(c)).cast("int")).alias(c))
for c in str_cols:
    aggs.append(ssum((col(c).isNull() | (trim(col(c)) == "")).cast("int")).alias(c))
for c in other_cols:
    aggs.append(ssum(col(c).isNull().cast("int")).alias(c))

print("Jumlah nilai hilang per kolom:")
train_df.agg(*aggs).show(truncate=False)

if num_cols:
    train_df.select(
        *[smin(c).alias(f"{c}_min") for c in num_cols],
        *[smax(c).alias(f"{c}_max") for c in num_cols]
    ).show(truncate=False)
    train_df.select(
        *[avg(c).alias(f"{c}_avg") for c in num_cols],
        *[stddev(c).alias(f"{c}_std") for c in num_cols]
    ).show(truncate=False)

if "passenger_count" in [c for c,_ in train_df.dtypes]:
    train_df.groupBy("passenger_count").agg(scount("*").alias("n")).orderBy("passenger_count").show()

# ========= 3) Preprocessing & Feature Engineering =========
n_before = train_df.count()

filtered = train_df.where(
    (col("pickup_longitude").between(-79.0, -71.0)) &
    (col("dropoff_longitude").between(-79.0, -71.0)) &
    (col("pickup_latitude").between(38.0, 45.0)) &
    (col("dropoff_latitude").between(38.0, 45.0)) &
    (col("fare_amount") >= 0) &
    (col("passenger_count").between(1, 6)) &
    col("pickup_datetime_ts").isNotNull()
)

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

fe.cache()
n_after = fe.count()
print(f"Baris sebelum filter: {n_before} | sesudah filter: {n_after} | dibuang: {n_before - n_after}")

# contoh sampling 1 juta baris
fe_small = fe.limit(1_000_000).cache()
print("Jumlah baris fe_small:", fe_small.count())


# ========= 4) Vectorization & Split =========
feature_cols = ["pickup_hour","pickup_dow","pickup_month","pickup_year","passenger_count","distance_km"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(withMean=True, withStd=True, inputCol="features_raw", outputCol="features")

prep_pipeline = Pipeline(stages=[assembler, scaler])
prep_model = prep_pipeline.fit(fe_small)
ds = prep_model.transform(fe_small).select("label","features")

train, test = ds.randomSplit([0.8, 0.2], seed=42)
print("Train:", train.count(), " Test:", test.count())

# ========= 5) Modeling =========
lr = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction")
lr_model = lr.fit(train)
pred_lr = lr_model.transform(test)

rf = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                           numTrees=120, maxDepth=14, seed=42)
rf_model = rf.fit(train)
pred_rf = rf_model.transform(test)

gbt = GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                   maxIter=120, maxDepth=8, seed=42)
gbt_model = gbt.fit(train)
pred_gbt = gbt_model.transform(test)

# ========= 6) Evaluasi =========
e_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
e_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
e_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

def report_metrics(name, pred):
    rmse = e_rmse.evaluate(pred)
    mae  = e_mae.evaluate(pred)
    r2   = e_r2.evaluate(pred)
    print(f"{name:8s} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

report_metrics("Linear", pred_lr)
report_metrics("RF",     pred_rf)
report_metrics("GBT",    pred_gbt)

# ========= 7) Tuning =========
rf_base = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction", seed=42)
paramGrid = (ParamGridBuilder()
             .addGrid(rf_base.numTrees, [80, 120, 160])
             .addGrid(rf_base.maxDepth, [10, 14, 18])
             .addGrid(rf_base.featureSubsetStrategy, ["auto","sqrt"])
             .build())

cv = CrossValidator(
    estimator=rf_base,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    numFolds=3,
    parallelism=2,
    seed=42
)

cv_model = cv.fit(train)
pred_cv = cv_model.transform(test)
print("Hasil model terbaik setelah tuning:")
report_metrics("RF-CV", pred_cv)

best_rf = cv_model.bestModel
print("Best params:",
      "numTrees=", best_rf.getNumTrees,
      "maxDepth=", best_rf.getOrDefault("maxDepth"),
      "featureSubsetStrategy=", best_rf.getOrDefault("featureSubsetStrategy"))

# ========= 8) Feature Importance & contoh prediksi =========
def show_feature_importance(model, feature_names):
    if hasattr(model, "featureImportances"):
        pairs = list(zip(feature_names, model.featureImportances))
        pairs_sorted = sorted(pairs, key=lambda x: float(x[1]), reverse=True)
        print("Feature importance:")
        for k, v in pairs_sorted:
            print(f"  {k:20s} -> {float(v):.6f}")
    else:
        print("Model tidak menyediakan featureImportances.")

print("\n=== Feature importance untuk RF terbaik ===")
show_feature_importance(best_rf, feature_cols)

print("\n=== Contoh prediksi vs label ===")
pred_cv.select("label","prediction").show(10, truncate=False)

spark.stop()
