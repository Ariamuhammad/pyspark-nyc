# importance.py
import os
from spark_utils import spark_session
from config import BEST_RF_DIR, FEATURE_COLS
from pyspark.ml.regression import RandomForestRegressionModel

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Feature Importance")

    if not os.path.exists(BEST_RF_DIR):
        raise FileNotFoundError("Best RF belum ada. Jalankan tuning.py dulu.")

    best = RandomForestRegressionModel.load(BEST_RF_DIR)
    fi = list(zip(FEATURE_COLS, [float(x) for x in best.featureImportances]))
    for k, v in sorted(fi, key=lambda x: x[1], reverse=True):
        print(f"{k:20s} -> {v:.6f}")

    spark.stop()
