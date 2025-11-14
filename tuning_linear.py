import os
from spark_utils import spark_session
from config import TRAIN_FP, TEST_FP, ART_DIR
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Tuning Linear Regression")

    if not (os.path.exists(TRAIN_FP) and os.path.exists(TEST_FP)):
        raise FileNotFoundError("Train/test parquet belum ada. Jalankan vectorize.py dulu.")

    train = spark.read.parquet(TRAIN_FP)
    test  = spark.read.parquet(TEST_FP)

    # Linear Regression dengan parameter yang bisa di-tune
    lr = LinearRegression(
        featuresCol="features", 
        labelCol="label", 
        predictionCol="prediction"
    )

    # Parameter grid untuk Linear Regression
    # Parameter yang bisa di-tune:
    # - regParam: Regularization parameter (L2 penalty)
    # - elasticNetParam: 0.0 = L2 (Ridge), 1.0 = L1 (Lasso), 0.5 = ElasticNet
    # - maxIter: Maximum iterations
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.0, 0.01, 0.1, 1.0])           # 4 values
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])          # 3 values
                 .build())
    # Total combinations: 4 × 3 = 12 models

    print(f"Total kombinasi parameter: {len(paramGrid)}")
    print("Parameter yang di-tune:")
    print("  - regParam: [0.0, 0.01, 0.1, 1.0]")
    print("  - elasticNetParam: [0.0 (Ridge), 0.5 (ElasticNet), 1.0 (Lasso)]")

    evaluator = RegressionEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="rmse"
    )

    # Cross-validator dengan 3-fold
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,   # Parallel processing untuk speed up
        seed=42
    )

    print("\nMemulai cross-validation (3-fold)...")
    print("Estimasi waktu: 5-10 menit")
    
    cv_model = cv.fit(train)
    
    # Best model
    best_lr = cv_model.bestModel
    
    # Get best parameters
    best_regParam = best_lr.getRegParam()
    best_elasticNet = best_lr.getElasticNetParam()
    
    print("\n" + "="*60)
    print("HASIL HYPERPARAMETER TUNING - LINEAR REGRESSION")
    print("="*60)
    print(f"Best regParam: {best_regParam}")
    print(f"Best elasticNetParam: {best_elasticNet}")
    
    if best_elasticNet == 0.0:
        reg_type = "Ridge (L2)"
    elif best_elasticNet == 1.0:
        reg_type = "Lasso (L1)"
    else:
        reg_type = "ElasticNet (L1+L2)"
    print(f"Regularization type: {reg_type}")
    
    # Evaluate on test set
    pred_test = best_lr.transform(test)
    
    e_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    e_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    e_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    
    rmse = e_rmse.evaluate(pred_test)
    mae = e_mae.evaluate(pred_test)
    r2 = e_r2.evaluate(pred_test)
    
    print("\nPerforma Model Terbaik (Test Set):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Compare with baseline
    print("\nPerbandingan dengan Baseline:")
    baseline_rmse = 5.8078
    baseline_mae = 2.4809
    baseline_r2 = 0.5377
    
    improvement_rmse = ((baseline_rmse - rmse) / baseline_rmse) * 100
    improvement_mae = ((baseline_mae - mae) / baseline_mae) * 100
    improvement_r2 = ((r2 - baseline_r2) / baseline_r2) * 100
    
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  Tuned RMSE:    {rmse:.4f}")
    print(f"  Improvement:   {improvement_rmse:+.2f}%")
    print()
    print(f"  Baseline MAE:  {baseline_mae:.4f}")
    print(f"  Tuned MAE:     {mae:.4f}")
    print(f"  Improvement:   {improvement_mae:+.2f}%")
    print()
    print(f"  Baseline R²:   {baseline_r2:.4f}")
    print(f"  Tuned R²:      {r2:.4f}")
    print(f"  Improvement:   {improvement_r2:+.2f}%")
    
    # Save best model
    BEST_LR_DIR = os.path.join(ART_DIR, "best_lr_model")
    best_lr.write().overwrite().save(BEST_LR_DIR)
    print(f"\nBest Linear Regression model tersimpan di: {BEST_LR_DIR}")
    print("="*60)

    spark.stop()
