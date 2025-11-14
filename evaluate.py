import os
from spark_utils import spark_session
from config import TEST_FP, BASELINE_DIR
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Evaluate Baselines")

    if not os.path.exists(TEST_FP):
        raise FileNotFoundError("Test parquet tidak ada. Jalankan vectorize.py dulu.")

    test = spark.read.parquet(TEST_FP)

    lr  = LinearRegressionModel.load(os.path.join(BASELINE_DIR,"lr"))
    rf  = RandomForestRegressionModel.load(os.path.join(BASELINE_DIR,"rf"))
    gbt = GBTRegressionModel.load(os.path.join(BASELINE_DIR,"gbt"))

    e_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    e_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    e_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    for name, model in [("Linear", lr), ("RF", rf), ("GBT", gbt)]:
        pred = model.transform(test)
        print(f"{name:6s} RMSE={e_rmse.evaluate(pred):.4f}  MAE={e_mae.evaluate(pred):.4f}  R2={e_r2.evaluate(pred):.4f}")

    spark.stop()
