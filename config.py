import os

# Path input
DATA_CSV = os.path.join("data", "train.csv")

# Folder artefak
ART_DIR = "artifacts"
EDA_DIR = os.path.join(ART_DIR, "eda")
RAW_PARQUET = os.path.join(ART_DIR, "raw.parquet")          # output load_data
FE_PARQUET  = os.path.join(ART_DIR, "fe.parquet")           # output preprocessing (full hasil filter+fitur)
FE_SMALL    = os.path.join(ART_DIR, "fe_small.parquet")     # subset agar ringan di laptop

# Folder preprocessing & modeling
PREP_DIR = os.path.join(ART_DIR, "prep_pipeline")          # pipeline vectorizer+scaler
TRAIN_FP = os.path.join(ART_DIR, "train.parquet")          # data training
TEST_FP  = os.path.join(ART_DIR, "test.parquet")           # data testing
BASELINE_DIR = os.path.join(ART_DIR, "baseline_models")    # model baseline
BEST_RF_DIR = os.path.join(ART_DIR, "best_rf_model")       # best model after tuning
METRICS_JSON = os.path.join(ART_DIR, "baseline_metrics.json")  # metrik evaluasi

# Kolom fitur yang akan dipakai modeling di tahap berikutnya
FEATURE_COLS = [
    "pickup_hour", "pickup_dow", "pickup_month", "pickup_year",
    "passenger_count", "distance_km"
]

# Batas penumpang yang dianggap valid secara bisnis
PASSENGER_MIN = 1
PASSENGER_MAX = 6

# Default batas visualisasi / analisis EDA (bisa di-override via env EDA_*)
EDA_FARE_MAX = 125.0
EDA_DISTANCE_MAX = 40.0
EDA_PICKUP_LON_MIN = -74.3
EDA_PICKUP_LON_MAX = -73.4
EDA_PICKUP_LAT_MIN = 40.4
EDA_PICKUP_LAT_MAX = 41.0
EDA_DROPOFF_LON_MIN = -74.3
EDA_DROPOFF_LON_MAX = -73.4
EDA_DROPOFF_LAT_MIN = 40.4
EDA_DROPOFF_LAT_MAX = 41.0
