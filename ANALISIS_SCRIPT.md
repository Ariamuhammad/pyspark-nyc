# 📊 ANALISIS SCRIPT NYC TAXI FARE PREDICTION

## ✅ KESIMPULAN UTAMA

### **Tujuan Dataset: BENAR** ✓

Dataset NYC Taxi Fare digunakan untuk **memprediksi tarif taksi** berdasarkan:

- 📍 Lokasi pickup & dropoff (longitude, latitude)
- ⏰ Waktu pickup (datetime)
- 👥 Jumlah penumpang
- 📏 Jarak perjalanan (dihitung dengan Haversine formula)

Target prediksi: `fare_amount` (tarif dalam USD)

---

## 🔍 EVALUASI SCRIPT `load_data.py`

### ✅ **KEKUATAN SCRIPT**

#### 1. **Schema Definition yang Tepat**

```python
schema = StructType([
    StructField("key", StringType(), True),
    StructField("fare_amount", DoubleType(), True),          # Target variable
    StructField("pickup_datetime", StringType(), True),       # Waktu
    StructField("pickup_longitude", DoubleType(), True),      # Koordinat
    StructField("pickup_latitude", DoubleType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("passenger_count", IntegerType(), True),      # Fitur
])
```

✓ Tipe data sesuai dengan karakteristik NYC Taxi dataset
✓ Mencegah error parsing saat membaca CSV

#### 2. **Data Quality Validation yang Komprehensif**

```python
def report_data_quality(df):
    geo_valid = (
        F.col("pickup_longitude").between(-79.0, -71.0) &    # Batas NY state
        F.col("dropoff_longitude").between(-79.0, -71.0) &
        F.col("pickup_latitude").between(38.0, 45.0) &       # Batas wajar
        F.col("dropoff_latitude").between(38.0, 45.0)
    )
```

**Validasi yang dilakukan:**

- ✓ Koordinat geografis dalam batas wilayah NYC/NY state
- ✓ Fare amount tidak negatif
- ✓ Passenger count dalam rentang valid (1-6)
- ✓ Missing values per kolom
- ✓ Deteksi duplikat berdasarkan key

#### 3. **Timestamp Conversion**

```python
.withColumn("pickup_datetime_ts", F.to_timestamp(F.col("pickup_datetime")))
```

✓ Konversi string ke timestamp untuk feature engineering
✓ Memungkinkan ekstraksi hour, day, month, year

#### 4. **Duplicate Handling**

```python
distinct_keys = df.select("key").distinct().count()
duplicate_rows = total_rows_initial - distinct_keys
if duplicate_rows > 0:
    df = df.dropDuplicates(["key"])
```

✓ Menghapus data duplikat berdasarkan primary key

#### 5. **Output ke Parquet**

```python
df.write.mode("overwrite").parquet(RAW_PARQUET)
```

✓ Format kolumnar yang efisien untuk Big Data
✓ Kompresi otomatis
✓ Lebih cepat dibaca daripada CSV

---

### ⚠️ **PERBAIKAN YANG SUDAH DILAKUKAN**

#### 1. **Typo di Schema Definition** [FIXED ✓]

**Sebelum:**

```python
schema = StructSchema = StructType([  # ← Typo!
```

**Sesudah:**

```python
schema = StructType([  # ✓ Benar
```

#### 2. **Config Paths yang Kurang Lengkap** [FIXED ✓]

**Ditambahkan ke `config.py`:**

```python
PREP_DIR = os.path.join(ART_DIR, "prep_pipeline")
TRAIN_FP = os.path.join(ART_DIR, "train.parquet")
TEST_FP  = os.path.join(ART_DIR, "test.parquet")
BASELINE_DIR = os.path.join(ART_DIR, "baseline_models")
METRICS_JSON = os.path.join(ART_DIR, "baseline_metrics.json")
```

---

## 🎯 ALUR PIPELINE LENGKAP

```
┌─────────────────┐
│  1. LOAD DATA   │  ← load_data.py (SUDAH BENAR ✓)
│  - Read CSV     │
│  - Validate     │
│  - Save Parquet │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. EDA         │  ← eda.py
│  - Statistics   │
│  - Correlations │
│  - Visualizations│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. PREPROCESS  │  ← preprocessing.py
│  - Filter data  │
│  - Calculate    │
│    distance     │
│  - Time features│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. VECTORIZE   │  ← vectorize.py
│  - Assemble     │
│  - Scale        │
│  - Train/Test   │
│    split        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. MODELING    │  ← modelling.py
│  - Linear Reg   │
│  - Random Forest│
│  - GBT          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. TUNING      │  ← tuning.py (jika ada)
│  - Cross Val    │
│  - Hyperparams  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. EVALUATE    │  ← evaluate.py
│  - RMSE, MAE   │
│  - R² Score    │
└─────────────────┘
```

---

## 📈 FITUR-FITUR YANG DIGUNAKAN

### **Input Features (FEATURE_COLS):**

```python
FEATURE_COLS = [
    "pickup_hour",      # Jam (0-23) - peak hours vs off-peak
    "pickup_dow",       # Day of week (1-7) - weekday vs weekend
    "pickup_month",     # Bulan (1-12) - seasonal patterns
    "pickup_year",      # Tahun - tren tahunan
    "passenger_count",  # Jumlah penumpang (1-6)
    "distance_km"       # Jarak Haversine (km) - FITUR PALING PENTING
]
```

### **Target Variable:**

```python
label = "fare_amount"  # Tarif taksi dalam USD
```

---

## 🧮 FEATURE ENGINEERING

### **1. Haversine Distance (preprocessing.py)**

```python
R = lit(6371.0)  # Radius bumi dalam km
phi1 = radians(col("pickup_latitude"))
phi2 = radians(col("dropoff_latitude"))
dphi = radians(col("dropoff_latitude") - col("pickup_latitude"))
dlmb = radians(col("dropoff_longitude") - col("pickup_longitude"))

a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*(sin(dlmb/2)**2)
c = 2*atan2(sqrt(a), sqrt(1-a))
dist_km = R*c
```

**Alasan:** Jarak adalah prediktor terkuat untuk tarif taksi

### **2. Time Features**

```python
hour(col("pickup_datetime_ts")).alias("pickup_hour")      # Rush hour effect
dayofweek(col("pickup_datetime_ts")).alias("pickup_dow")  # Weekend premium
month(col("pickup_datetime_ts")).alias("pickup_month")    # Seasonal demand
year(col("pickup_datetime_ts")).alias("pickup_year")      # Price trends
```

---

## 🎓 KESESUAIAN UNTUK TUGAS BIG DATA & AI

### ✅ **Aspek Big Data:**

- ✓ Menggunakan **PySpark** (distributed processing)
- ✓ Format **Parquet** (columnar storage)
- ✓ Lazy evaluation & caching
- ✓ Scalable untuk dataset besar (5M+ rows)

### ✅ **Aspek AI/ML:**

- ✓ Feature engineering yang tepat
- ✓ Multiple algorithms (Linear Reg, RF, GBT)
- ✓ Standardization/Scaling
- ✓ Train/test split
- ✓ Cross-validation
- ✓ Evaluation metrics (RMSE, MAE, R²)

---

## 📊 DATA QUALITY CHECKS

### **Validasi yang Dilakukan:**

| Check              | Range/Condition                | Purpose               |
| ------------------ | ------------------------------ | --------------------- |
| **Geografis**      | Lon: -79 to -71, Lat: 38 to 45 | Wilayah NY state      |
| **Fare**           | ≥ 0                            | Tidak boleh negatif   |
| **Passenger**      | 1-6                            | Kapasitas normal taxi |
| **Timestamp**      | Not NULL                       | Untuk time features   |
| **Duplicates**     | Unique key                     | Data integrity        |
| **Missing Values** | Count per column               | Completeness          |

---

## 🚀 CARA MENJALANKAN

```bash
# 1. Load & validate raw data
python load_data.py

# 2. Exploratory data analysis
python eda.py

# 3. Feature engineering
python preprocessing.py

# 4. Prepare for modeling
python vectorize.py

# 5. Train baseline models
python modelling.py

# 6. Hyperparameter tuning (optional)
python tuning.py

# 7. Final evaluation
python evaluate.py
```

---

## 📝 REKOMENDASI TAMBAHAN

### **1. Tambahkan Error Handling**

```python
try:
    df = spark.read.csv(DATA_CSV, header=True, schema=schema)
except Exception as e:
    print(f"Error reading CSV: {e}")
    raise
```

### **2. Logging yang Lebih Baik**

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Loaded {total_rows:,} rows from {DATA_CSV}")
logger.warning(f"Dropped {duplicate_rows:,} duplicate rows")
```

### **3. Config Validation**

```python
assert PASSENGER_MIN < PASSENGER_MAX, "Invalid passenger range"
assert os.path.exists("data"), "Data folder not found"
```

### **4. Data Versioning**

```python
# Tambahkan timestamp ke output
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{RAW_PARQUET}_{timestamp}"
```

---

## ✅ KESIMPULAN AKHIR

### **Script `load_data.py` SUDAH BAIK dan SESUAI untuk:**

✓ **Preprocessing data NYC Taxi Fare**  
✓ **Data quality validation**  
✓ **Persiapan untuk modeling**  
✓ **Tugas Big Data & AI**

### **Perbaikan yang Sudah Dilakukan:**

✓ Typo `StructSchema` → `StructType`  
✓ Config paths lengkap untuk semua tahap pipeline

### **Status:**

🟢 **READY TO USE** - Script siap dijalankan!

---

**Dibuat oleh:** GitHub Copilot  
**Tanggal:** November 9, 2025  
**Project:** NYC Taxi Fare Prediction with PySpark
