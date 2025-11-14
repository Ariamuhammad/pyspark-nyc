import os
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from spark_utils import spark_session
from config import FE_SMALL, FEATURE_COLS, PREP_DIR, TRAIN_FP, TEST_FP

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Vectorize")

    if not os.path.exists(FE_SMALL):
        raise FileNotFoundError("Tidak menemukan artifacts/fe_small.parquet. Jalankan preprocessing dulu.")

    fe_small = spark.read.parquet(FE_SMALL)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler = StandardScaler(withMean=True, withStd=True, inputCol="features_raw", outputCol="features")
    prep = Pipeline(stages=[assembler, scaler])
    prep_model = prep.fit(fe_small)

    ds = prep_model.transform(fe_small).select("label", "features")
    train, test = ds.randomSplit([0.8, 0.2], seed=42)
    print("Train:", train.count(), " Test:", test.count())

    prep_model.write().overwrite().save(PREP_DIR)
    train.write.mode("overwrite").parquet(TRAIN_FP)
    test.write.mode("overwrite").parquet(TEST_FP)
    print(f"Tersimpan: {PREP_DIR}, {TRAIN_FP}, {TEST_FP}")

    spark.stop()
