# modeling.py
import os, json
from spark_utils import spark_session
from config import TRAIN_FP, TEST_FP, BASELINE_DIR, METRICS_JSON
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Baseline Models")
    os.makedirs(BASELINE_DIR, exist_ok=True)

    if not (os.path.exists(TRAIN_FP) and os.path.exists(TEST_FP)):
        raise FileNotFoundError("Train/test parquet belum ada. Jalankan vectorize.py dulu.")

    train = spark.read.parquet(TRAIN_FP)
    test  = spark.read.parquet(TEST_FP)

    e_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    e_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    e_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    # Baseline LR dengan regularisasi kecil agar stabil
    lr = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction",
                          regParam=0.01, elasticNetParam=0.0)
    lr_model = lr.fit(train)
    pred_lr = lr_model.transform(test)

    # Random Forest baseline
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                               numTrees=120, maxDepth=14, seed=42)
    rf_model = rf.fit(train)
    pred_rf = rf_model.transform(test)

    # Gradient Boosted Trees baseline
    gbt = GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                       maxIter=120, maxDepth=8, seed=42)
    gbt_model = gbt.fit(train)
    pred_gbt = gbt_model.transform(test)

    metrics = {
        "Linear": {"rmse": e_rmse.evaluate(pred_lr), "mae": e_mae.evaluate(pred_lr), "r2": e_r2.evaluate(pred_lr)},
        "RF":     {"rmse": e_rmse.evaluate(pred_rf), "mae": e_mae.evaluate(pred_rf), "r2": e_r2.evaluate(pred_rf)},
        "GBT":    {"rmse": e_rmse.evaluate(pred_gbt), "mae": e_mae.evaluate(pred_gbt), "r2": e_r2.evaluate(pred_gbt)},
    }

    print(json.dumps(metrics, indent=2))

    lr_model.save(os.path.join(BASELINE_DIR, "lr"))
    rf_model.save(os.path.join(BASELINE_DIR, "rf"))
    gbt_model.save(os.path.join(BASELINE_DIR, "gbt"))

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model baseline tersimpan di {BASELINE_DIR}")
    print(f"Metrik baseline tersimpan di {METRICS_JSON}")
    spark.stop()
