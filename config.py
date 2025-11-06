import os

# Path input
DATA_CSV = os.path.join("data", "train.csv")

# Folder artefak
ART_DIR = "artifacts"
RAW_PARQUET = os.path.join(ART_DIR, "raw.parquet")          # output load_data
FE_PARQUET  = os.path.join(ART_DIR, "fe.parquet")           # output preprocessing (full hasil filter+fitur)
FE_SMALL    = os.path.join(ART_DIR, "fe_small.parquet")     # subset agar ringan di laptop

# Kolom fitur yang akan dipakai modeling di tahap berikutnya
FEATURE_COLS = [
    "pickup_hour", "pickup_dow", "pickup_month", "pickup_year",
    "passenger_count", "distance_km"
]
