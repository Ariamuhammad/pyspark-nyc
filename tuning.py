import os
from spark_utils import spark_session
from config import TRAIN_FP, TEST_FP, BEST_RF_DIR
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Tuning RF")

    if not (os.path.exists(TRAIN_FP) and os.path.exists(TEST_FP)):
        raise FileNotFoundError("Train/test parquet belum ada. Jalankan vectorize.py dulu.")

    train = spark.read.parquet(TRAIN_FP)
    test  = spark.read.parquet(TEST_FP)

    rf = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction", seed=42)

    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [80])         # 1 nilai
                 .addGrid(rf.maxDepth, [10, 14])     # 2 nilai
                 .build())

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=1,
        seed=42
    )

    cv_model = cv.fit(train)
    pred_cv = cv_model.transform(test)

    rmse = evaluator.evaluate(pred_cv)
    print(f"RF-CV RMSE={rmse:.4f}")

    cv_model.bestModel.write().overwrite().save(BEST_RF_DIR)
    print(f"Best RF tersimpan di {BEST_RF_DIR}")

    spark.stop()
