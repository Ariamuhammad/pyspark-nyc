# predictions.py
"""
Generate prediction comparison table: Actual vs Predicted Fare
Output: CSV dengan sample predictions untuk analisis & presentasi
"""
import os
import json
from spark_utils import spark_session
from config import BEST_RF_DIR, TEST_FP, ART_DIR
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql.functions import col, abs as spark_abs, round as spark_round

if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - Predictions")

    # Load best model
    if not os.path.exists(BEST_RF_DIR):
        print(f"⚠️  Best RF model not found at {BEST_RF_DIR}")
        print("   Using baseline RF model instead...")
        from config import BASELINE_DIR
        model_path = os.path.join(BASELINE_DIR, "rf")
    else:
        model_path = BEST_RF_DIR

    print(f"Loading model from: {model_path}")
    best_model = RandomForestRegressionModel.load(model_path)

    # Load test data
    if not os.path.exists(TEST_FP):
        raise FileNotFoundError(f"Test data not found: {TEST_FP}. Run vectorize.py first.")

    test = spark.read.parquet(TEST_FP)
    print(f"Test data loaded: {test.count()} rows")

    # Make predictions
    print("\nMaking predictions...")
    predictions = best_model.transform(test)

    # Calculate error metrics
    predictions = predictions.withColumn(
        "error", 
        col("prediction") - col("label")
    )
    predictions = predictions.withColumn(
        "absolute_error", 
        spark_abs(col("error"))
    )
    predictions = predictions.withColumn(
        "percentage_error",
        (spark_abs(col("error")) / col("label")) * 100
    )

    # Round for readability
    predictions = predictions.withColumn("label", spark_round(col("label"), 2))
    predictions = predictions.withColumn("prediction", spark_round(col("prediction"), 2))
    predictions = predictions.withColumn("error", spark_round(col("error"), 2))
    predictions = predictions.withColumn("absolute_error", spark_round(col("absolute_error"), 2))
    predictions = predictions.withColumn("percentage_error", spark_round(col("percentage_error"), 2))

    # Select relevant columns
    result = predictions.select(
        "label",
        "prediction",
        "error",
        "absolute_error",
        "percentage_error"
    )

    # Summary statistics
    print("\n" + "="*70)
    print("PREDICTION SUMMARY STATISTICS")
    print("="*70)
    
    summary = result.select(
        spark_abs(col("error")).alias("abs_error"),
        col("percentage_error")
    ).describe()
    
    summary.show()

    # Show sample predictions
    # print("\n" + "="*70)
    # print("SAMPLE PREDICTIONS (First 20 rows)")
    # print("="*70)
    # result.show(20, truncate=False)

    # Get best predictions (lowest error)
    print("\n" + "="*70)
    print("BEST PREDICTIONS (Lowest Absolute Error)")
    print("="*70)
    result.orderBy("absolute_error").show(10, truncate=False)

    # Get worst predictions (highest error)
    print("\n" + "="*70)
    print("WORST PREDICTIONS (Highest Absolute Error)")
    print("="*70)
    result.orderBy(col("absolute_error").desc()).show(10, truncate=False)

    # Save full predictions
    predictions_dir = os.path.join(ART_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Save full predictions to parquet
    full_output = os.path.join(predictions_dir, "predictions_full.parquet")
    result.write.mode("overwrite").parquet(full_output)
    print(f"\n✅ Full predictions saved: {full_output}")

    # Save sample to CSV for easy viewing
    sample_size = min(1000, result.count())
    csv_output = os.path.join(predictions_dir, "predictions_sample.csv")
    result.limit(sample_size).toPandas().to_csv(csv_output, index=False)
    print(f"✅ Sample predictions (first {sample_size}) saved: {csv_output}")

    # Save best/worst predictions to CSV
    best_csv = os.path.join(predictions_dir, "predictions_best.csv")
    result.orderBy("absolute_error").limit(100).toPandas().to_csv(best_csv, index=False)
    print(f"✅ Best 100 predictions saved: {best_csv}")

    worst_csv = os.path.join(predictions_dir, "predictions_worst.csv")
    result.orderBy(col("absolute_error").desc()).limit(100).toPandas().to_csv(worst_csv, index=False)
    print(f"✅ Worst 100 predictions saved: {worst_csv}")

    # Error distribution analysis
    print("\n" + "="*70)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("="*70)

    # Categorize errors
    error_categories = result.selectExpr(
        "CASE " +
        "WHEN absolute_error < 1.0 THEN 'Excellent (<$1)' " +
        "WHEN absolute_error < 2.0 THEN 'Good ($1-$2)' " +
        "WHEN absolute_error < 5.0 THEN 'Fair ($2-$5)' " +
        "WHEN absolute_error < 10.0 THEN 'Poor ($5-$10)' " +
        "ELSE 'Very Poor (>$10)' " +
        "END as error_category"
    )

    error_dist = error_categories.groupBy("error_category").count()
    error_dist = error_dist.withColumn(
        "percentage",
        spark_round((col("count") / result.count()) * 100, 2)
    )
    error_dist.orderBy("count", ascending=False).show(truncate=False)

    # Save error distribution
    error_dist_csv = os.path.join(predictions_dir, "error_distribution.csv")
    error_dist.toPandas().to_csv(error_dist_csv, index=False)
    print(f"✅ Error distribution saved: {error_dist_csv}")

    # Create summary JSON
    summary_stats = {
        "total_predictions": result.count(),
        "mean_absolute_error": float(result.select(spark_abs(col("error"))).agg({"abs(error)": "avg"}).collect()[0][0]),
        "median_absolute_error": float(result.approxQuantile("absolute_error", [0.5], 0.01)[0]),
        "max_error": float(result.select("absolute_error").agg({"absolute_error": "max"}).collect()[0][0]),
        "min_error": float(result.select("absolute_error").agg({"absolute_error": "min"}).collect()[0][0]),
        "mean_percentage_error": float(result.select("percentage_error").agg({"percentage_error": "avg"}).collect()[0][0]),
        "predictions_within_1_dollar": int(result.filter(col("absolute_error") < 1.0).count()),
        "predictions_within_2_dollars": int(result.filter(col("absolute_error") < 2.0).count()),
        "predictions_within_5_dollars": int(result.filter(col("absolute_error") < 5.0).count()),
    }

    # Add percentages
    total = summary_stats["total_predictions"]
    summary_stats["pct_within_1_dollar"] = round((summary_stats["predictions_within_1_dollar"] / total) * 100, 2)
    summary_stats["pct_within_2_dollars"] = round((summary_stats["predictions_within_2_dollars"] / total) * 100, 2)
    summary_stats["pct_within_5_dollars"] = round((summary_stats["predictions_within_5_dollars"] / total) * 100, 2)

    summary_json = os.path.join(predictions_dir, "predictions_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"✅ Prediction summary saved: {summary_json}")

    print("\n" + "="*70)
    print("PREDICTION ACCURACY BREAKDOWN")
    print("="*70)
    print(f"Within $1:  {summary_stats['predictions_within_1_dollar']:,} ({summary_stats['pct_within_1_dollar']}%)")
    print(f"Within $2:  {summary_stats['predictions_within_2_dollars']:,} ({summary_stats['pct_within_2_dollars']}%)")
    print(f"Within $5:  {summary_stats['predictions_within_5_dollars']:,} ({summary_stats['pct_within_5_dollars']}%)")
    print("="*70)

    print("\n✅ Prediction analysis complete!")
    print(f"\nAll outputs saved in: {predictions_dir}/")
    print("  - predictions_full.parquet       (all predictions)")
    print("  - predictions_sample.csv         (first 1000 rows)")
    print("  - predictions_best.csv           (best 100 predictions)")
    print("  - predictions_worst.csv          (worst 100 predictions)")
    print("  - error_distribution.csv         (error categories)")
    print("  - predictions_summary.json       (summary statistics)")

    spark.stop()
