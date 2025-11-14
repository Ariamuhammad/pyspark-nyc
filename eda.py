import math
import os
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.functions import (
    avg,
    col,
    dayofweek,
    hour,
    isnan,
    lit,
    min as smin,
    max as smax,
    month,
    radians,
    sin,
    cos,
    sqrt,
    atan2,
    stddev,
    sum as ssum,
    trim,
    year,
)

from spark_utils import spark_session
from config import (
    RAW_PARQUET,
    EDA_DIR,
    PASSENGER_MIN,
    PASSENGER_MAX,
    EDA_FARE_MAX,
    EDA_DISTANCE_MAX,
    EDA_PICKUP_LON_MIN,
    EDA_PICKUP_LON_MAX,
    EDA_PICKUP_LAT_MIN,
    EDA_PICKUP_LAT_MAX,
    EDA_DROPOFF_LON_MIN,
    EDA_DROPOFF_LON_MAX,
    EDA_DROPOFF_LAT_MIN,
    EDA_DROPOFF_LAT_MAX,
)

# ----- Konstanta & path artefak -------------------------------------------------
FARE_BIN_WIDTH = float(os.environ.get("EDA_FARE_BIN_WIDTH", "1.0"))
DISTANCE_BIN_WIDTH = float(os.environ.get("EDA_DISTANCE_BIN_WIDTH", "0.5"))
GEO_BIN_SIZE = float(os.environ.get("EDA_GEO_BIN_SIZE", "0.01"))
HEAT_DISTANCE_BIN = float(os.environ.get("EDA_HEAT_DISTANCE_BIN", "0.5"))
HEAT_FARE_BIN = float(os.environ.get("EDA_HEAT_FARE_BIN", "2.5"))
HEAT_DISTANCE_MAX = float(os.environ.get("EDA_HEAT_DISTANCE_MAX", str(EDA_DISTANCE_MAX)))
HEAT_FARE_MAX = float(os.environ.get("EDA_HEAT_FARE_MAX", str(EDA_FARE_MAX)))

PICKUP_LON_MIN = float(os.environ.get("EDA_PICKUP_LON_MIN", str(EDA_PICKUP_LON_MIN)))
PICKUP_LON_MAX = float(os.environ.get("EDA_PICKUP_LON_MAX", str(EDA_PICKUP_LON_MAX)))
PICKUP_LAT_MIN = float(os.environ.get("EDA_PICKUP_LAT_MIN", str(EDA_PICKUP_LAT_MIN)))
PICKUP_LAT_MAX = float(os.environ.get("EDA_PICKUP_LAT_MAX", str(EDA_PICKUP_LAT_MAX)))
DROPOFF_LON_MIN = float(os.environ.get("EDA_DROPOFF_LON_MIN", str(EDA_DROPOFF_LON_MIN)))
DROPOFF_LON_MAX = float(os.environ.get("EDA_DROPOFF_LON_MAX", str(EDA_DROPOFF_LON_MAX)))
DROPOFF_LAT_MIN = float(os.environ.get("EDA_DROPOFF_LAT_MIN", str(EDA_DROPOFF_LAT_MIN)))
DROPOFF_LAT_MAX = float(os.environ.get("EDA_DROPOFF_LAT_MAX", str(EDA_DROPOFF_LAT_MAX)))
SCATTER_MAX_DISTANCE = float(os.environ.get("EDA_SCATTER_MAX_DISTANCE", str(EDA_DISTANCE_MAX)))
SCATTER_MAX_FARE = float(os.environ.get("EDA_SCATTER_MAX_FARE", str(EDA_FARE_MAX)))

PLOT_DIR = os.path.join(EDA_DIR, "plots")
MISSING_CSV = os.path.join(EDA_DIR, "missing_values.csv")
MIN_MAX_CSV = os.path.join(EDA_DIR, "numeric_min_max.csv")
AVG_STD_CSV = os.path.join(EDA_DIR, "numeric_avg_std.csv")
SUMMARY_TXT = os.path.join(EDA_DIR, "summary.txt")
FARE_HIST_CSV = os.path.join(EDA_DIR, "hist_fare_amount.csv")
DIST_HIST_CSV = os.path.join(EDA_DIR, "hist_distance_km.csv")
PASSENGER_CSV = os.path.join(EDA_DIR, "passenger_distribution.csv")
HOUR_CSV = os.path.join(EDA_DIR, "pickup_hour_distribution.csv")
MONTHLY_CSV = os.path.join(EDA_DIR, "trips_per_month.csv")
PICKUP_HEAT_CSV = os.path.join(EDA_DIR, "pickup_heatmap.csv")
DROPOFF_HEAT_CSV = os.path.join(EDA_DIR, "dropoff_heatmap.csv")
FARE_DISTANCE_HEAT_CSV = os.path.join(EDA_DIR, "fare_distance_heatmap.csv")
CORR_CSV = os.path.join(EDA_DIR, "correlation_matrix.csv")
EDA_DASHBOARD = os.path.join(PLOT_DIR, "eda_dashboard.png")
MISSING_PLOT = os.path.join(PLOT_DIR, "missing_values.png")


# ----- Helper -------------------------------------------------------------------
def add_distance_feature(df):
    """Tambahkan kolom jarak Haversine (km) agar siap dipakai di EDA & modeling."""
    R = lit(6371.0)
    phi1 = radians(col("pickup_latitude"))
    phi2 = radians(col("dropoff_latitude"))
    dphi = radians(col("dropoff_latitude") - col("pickup_latitude"))
    dlmb = radians(col("dropoff_longitude") - col("pickup_longitude"))
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * (sin(dlmb / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return df.withColumn("distance_km", R * c)


def compute_histogram(df, column, bin_width, min_override=None, max_override=None):
    stats = df.select(
        smin(column).alias("min_val"),
        smax(column).alias("max_val"),
    ).collect()[0]
    min_val = min_override if min_override is not None else float(stats["min_val"] or 0)
    max_val = max_override if max_override is not None else float(stats["max_val"] or 0)
    if max_val <= min_val:
        max_val = min_val + bin_width

    num_bins = max(1, int(math.ceil((max_val - min_val) / bin_width)))
    hist_df = (
        df.where(col(column).isNotNull())
        .withColumn(
            "bin_id",
            F.floor((col(column) - F.lit(min_val)) / bin_width).cast("int"),
        )
        .where((col("bin_id") >= 0) & (col("bin_id") <= num_bins))
        .groupBy("bin_id")
        .agg(F.count("*").alias("count"))
        .orderBy("bin_id")
    )
    pdf = hist_df.toPandas()
    if pdf.empty:
        return pdf
    pdf["bin_left"] = min_val + pdf["bin_id"] * bin_width
    pdf["bin_right"] = pdf["bin_left"] + bin_width
    pdf["bin_mid"] = (pdf["bin_left"] + pdf["bin_right"]) / 2
    return pdf


def compute_distribution(df, column):
    return (
        df.groupBy(column)
        .agg(F.count("*").alias("count"))
        .orderBy(column)
        .toPandas()
    )


def compute_passenger_distribution(df):
    return (
        df.where(col("passenger_count").between(PASSENGER_MIN, PASSENGER_MAX))
        .groupBy("passenger_count")
        .agg(F.count("*").alias("count"))
        .orderBy("passenger_count")
        .toPandas()
    )


def compute_monthly_trips(df):
    agg = (
        df.groupBy("pickup_year", "pickup_month")
        .agg(F.count("*").alias("count"))
        .orderBy("pickup_year", "pickup_month")
    )
    pdf = agg.toPandas()
    if pdf.empty:
        return pdf
    pdf["label"] = (
        pdf["pickup_year"].astype(int).astype(str)
        + "-"
        + pdf["pickup_month"].astype(int).astype(str).str.zfill(2)
    )
    return pdf


def compute_geo_heatmap(df, lon_col, lat_col, lon_min, lon_max, lat_min, lat_max):
    filtered = df.where(
        col(lon_col).between(lon_min, lon_max)
        & col(lat_col).between(lat_min, lat_max)
    )
    heat_df = (
        filtered.withColumn(
            "lon_bin", F.floor((col(lon_col) - lon_min) / GEO_BIN_SIZE).cast("int")
        )
        .withColumn(
            "lat_bin", F.floor((col(lat_col) - lat_min) / GEO_BIN_SIZE).cast("int")
        )
        .groupBy("lat_bin", "lon_bin")
        .agg(F.count("*").alias("count"))
    )
    pdf = heat_df.toPandas()
    if pdf.empty:
        return pdf
    pdf["lon_center"] = lon_min + (pdf["lon_bin"] + 0.5) * GEO_BIN_SIZE
    pdf["lat_center"] = lat_min + (pdf["lat_bin"] + 0.5) * GEO_BIN_SIZE
    return pdf


def pivot_heatmap(pdf, value_col="count"):
    if pdf.empty:
        return pd.DataFrame()
    pivot = (
        pdf.pivot(index="lat_bin", columns="lon_bin", values=value_col)
        .fillna(0)
        .sort_index(ascending=False)
    )
    lat_map = pdf.drop_duplicates("lat_bin").set_index("lat_bin")["lat_center"].to_dict()
    lon_map = pdf.drop_duplicates("lon_bin").set_index("lon_bin")["lon_center"].to_dict()
    pivot.index = pivot.index.map(lambda idx: round(lat_map.get(idx, idx), 4))
    pivot.columns = pivot.columns.map(lambda idx: round(lon_map.get(idx, idx), 4))
    return pivot


def compute_fare_distance_heatmap(df):
    filtered = df.where(
        (col("distance_km") >= 0)
        & (col("distance_km") <= HEAT_DISTANCE_MAX)
        & (col("fare_amount") >= 0)
        & (col("fare_amount") <= HEAT_FARE_MAX)
    )
    heat_df = (
        filtered.withColumn(
            "dist_bin", F.floor(col("distance_km") / HEAT_DISTANCE_BIN).cast("int")
        )
        .withColumn(
            "fare_bin", F.floor(col("fare_amount") / HEAT_FARE_BIN).cast("int")
        )
        .groupBy("fare_bin", "dist_bin")
        .agg(F.count("*").alias("count"))
    )
    pdf = heat_df.toPandas()
    if pdf.empty:
        return pdf
    pdf["dist_center"] = (pdf["dist_bin"] + 0.5) * HEAT_DISTANCE_BIN
    pdf["fare_center"] = (pdf["fare_bin"] + 0.5) * HEAT_FARE_BIN
    return pdf


def save_dataframe(pdf, path):
    if not pdf.empty:
        pdf.to_csv(path, index=False)


def plot_dashboard(
    fare_hist,
    distance_hist,
    passenger_dist,
    monthly_trips,
    hour_dist,
    pickup_heat,
    dropoff_heat,
    fare_distance_heat,
    corr_matrix,
):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        "NYC Taxi Fare Dataset - Exploratory Data Analysis",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    def safe_bar(ax, x, y, title, xlabel, ylabel, color="#1f77b4"):
        if len(x) == 0:
            ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
        else:
            ax.bar(x, y, color=color, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    # Row 0
    safe_bar(
        axes[0, 0],
        fare_hist.get("bin_mid", []),
        fare_hist.get("count", []),
        "Distribusi Fare (USD)",
        "Fare (bin midpoint)",
        "Jumlah perjalanan",
        color="#2c7fb8",
    )
    safe_bar(
        axes[0, 1],
        distance_hist.get("bin_mid", []),
        distance_hist.get("count", []),
        "Distribusi Distance (km)",
        "Distance (bin midpoint)",
        "Jumlah perjalanan",
        color="#41ab5d",
    )
    safe_bar(
        axes[0, 2],
        passenger_dist.get("passenger_count", []),
        passenger_dist.get("count", []),
        "Distribusi Passenger Count",
        "Passenger count",
        "Jumlah perjalanan",
        color="#fdae61",
    )

    # Row 1
    if not monthly_trips.empty:
        axes[1, 0].plot(
            monthly_trips["label"],
            monthly_trips["count"],
            marker="o",
            color="#3182bd",
        )
        axes[1, 0].set_xticks(monthly_trips["label"][:: max(1, len(monthly_trips) // 8)])
        axes[1, 0].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
    axes[1, 0].set_title("Tren Perjalanan per Bulan")
    axes[1, 0].set_xlabel("Tahun-Bulan")
    axes[1, 0].set_ylabel("Jumlah perjalanan")
    axes[1, 0].grid(True, alpha=0.3)

    safe_bar(
        axes[1, 1],
        hour_dist.get("pickup_hour", []),
        hour_dist.get("count", []),
        "Distribusi Pickup per Jam",
        "Jam",
        "Jumlah perjalanan",
        color="#756bb1",
    )

    if not pickup_heat.empty:
        sns.heatmap(
            pickup_heat,
            ax=axes[1, 2],
            cmap="magma",
            cbar_kws={"shrink": 0.7, "label": "Trips"},
        )
        axes[1, 2].invert_yaxis()
        axes[1, 2].set_title("Pickup Density (NYC bounds)")
        axes[1, 2].set_xlabel("Longitude")
        axes[1, 2].set_ylabel("Latitude")
    else:
        axes[1, 2].text(0.5, 0.5, "Tidak ada data pickup dalam batas", ha="center", va="center")

    # Row 2
    if not dropoff_heat.empty:
        sns.heatmap(
            dropoff_heat,
            ax=axes[2, 0],
            cmap="magma",
            cbar_kws={"shrink": 0.7, "label": "Trips"},
        )
        axes[2, 0].invert_yaxis()
        axes[2, 0].set_title("Dropoff Density (NYC bounds)")
        axes[2, 0].set_xlabel("Longitude")
        axes[2, 0].set_ylabel("Latitude")
    else:
        axes[2, 0].text(0.5, 0.5, "Tidak ada data dropoff dalam batas", ha="center", va="center")

    if not fare_distance_heat.empty:
        sns.heatmap(
            fare_distance_heat,
            ax=axes[2, 1],
            cmap="viridis",
            cbar_kws={"shrink": 0.7, "label": "Trips"},
        )
        axes[2, 1].invert_yaxis()
        axes[2, 1].set_title("Fare vs Distance Density")
        axes[2, 1].set_xlabel("Distance (km)")
        axes[2, 1].set_ylabel("Fare (USD)")
    else:
        axes[2, 1].text(0.5, 0.5, "Tidak ada data dalam rentang heatmap", ha="center", va="center")

    if not corr_matrix.empty:
        # Sesuaikan font size berdasarkan jumlah kolom
        annot_size = 10 if len(corr_matrix.columns) <= 7 else 8
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",  # 3 decimal untuk lebih presisi
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7, "label": "Correlation"},
            ax=axes[2, 2],
            annot_kws={"size": annot_size},
            vmin=-1, vmax=1  # Rentang correlation -1 to 1
        )
        axes[2, 2].set_title("Correlation Matrix (All Features)")
        axes[2, 2].set_xlabel("")
        axes[2, 2].set_ylabel("")
    else:
        axes[2, 2].text(0.5, 0.5, "Correlation tidak tersedia", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EDA_DASHBOARD, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values(missing_df):
    if missing_df.empty or missing_df["missing_count"].sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        missing_df["column"],
        missing_df["missing_count"],
        color="#fb6a4a",
        edgecolor="black",
    )
    ax.set_xticklabels(missing_df["column"], rotation=45, ha="right")
    ax.set_ylabel("Missing count")
    ax.set_title("Missing Values per Column")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(MISSING_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    spark = spark_session(app_name="NYC Taxi - EDA")
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = spark.read.parquet(RAW_PARQUET)
    df = add_distance_feature(df)
    df = (
        df.withColumn("pickup_hour", hour(col("pickup_datetime_ts")))
        .withColumn("pickup_dow", dayofweek(col("pickup_datetime_ts")))
        .withColumn("pickup_month", month(col("pickup_datetime_ts")))
        .withColumn("pickup_year", year(col("pickup_datetime_ts")))
    )
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")

    # Missing value summary
    numeric_types = {
        "double",
        "float",
        "int",
        "bigint",
        "long",
        "decimal",
        "smallint",
        "tinyint",
    }
    num_cols = [c for c, t in df.dtypes if t in numeric_types]
    str_cols = [c for c, t in df.dtypes if t == "string"]
    other_cols = [c for c, t in df.dtypes if (t not in numeric_types and t != "string")]

    aggs = []
    for c in num_cols:
        aggs.append(ssum((col(c).isNull() | isnan(c)).cast("int")).alias(c))
    for c in str_cols:
        aggs.append(ssum((col(c).isNull() | (trim(col(c)) == "")).cast("int")).alias(c))
    for c in other_cols:
        aggs.append(ssum(col(c).isNull().cast("int")).alias(c))

    missing_values_df = df.agg(*aggs)
    missing_values_pd = (
        missing_values_df.toPandas().T.reset_index().rename(columns={"index": "column", 0: "missing_count"})
    )
    missing_values_pd["missing_pct"] = (
        missing_values_pd["missing_count"] / total_rows * 100
    )
    save_dataframe(missing_values_pd, MISSING_CSV)
    plot_missing_values(missing_values_pd)

    # Numeric stats
    if num_cols:
        min_max_df = df.select(
            *[smin(c).alias(f"{c}_min") for c in num_cols],
            *[smax(c).alias(f"{c}_max") for c in num_cols],
        )
        avg_std_df = df.select(
            *[avg(c).alias(f"{c}_avg") for c in num_cols],
            *[stddev(c).alias(f"{c}_std") for c in num_cols],
        )
        min_max_df.toPandas().to_csv(MIN_MAX_CSV, index=False)
        avg_std_df.toPandas().to_csv(AVG_STD_CSV, index=False)

    # Spatial sanity check
    pickup_valid = col("pickup_longitude").between(PICKUP_LON_MIN, PICKUP_LON_MAX) & col(
        "pickup_latitude"
    ).between(PICKUP_LAT_MIN, PICKUP_LAT_MAX)
    dropoff_valid = col("dropoff_longitude").between(
        DROPOFF_LON_MIN, DROPOFF_LON_MAX
    ) & col("dropoff_latitude").between(DROPOFF_LAT_MIN, DROPOFF_LAT_MAX)
    pickup_outside = df.filter(~pickup_valid).count()
    dropoff_outside = df.filter(~dropoff_valid).count()

    # Outlier detection (IQR) untuk fare & distance
    def iqr_summary(column):
        q1, median, q3 = df.approxQuantile(column, [0.25, 0.5, 0.75], 0.01)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df.filter((col(column) < lower) | (col(column) > upper)).count()
        return {
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "outliers": outliers,
        }

    fare_iqr = iqr_summary("fare_amount")
    dist_iqr = iqr_summary("distance_km")

    # Distributions / aggregated data untuk visual
    fare_hist_pd = compute_histogram(df, "fare_amount", FARE_BIN_WIDTH, 0, HEAT_FARE_MAX)
    distance_hist_pd = compute_histogram(df, "distance_km", DISTANCE_BIN_WIDTH, 0, HEAT_DISTANCE_MAX)
    passenger_dist_pd = compute_passenger_distribution(df)
    hour_dist_pd = compute_distribution(df, "pickup_hour")
    monthly_trips_pd = compute_monthly_trips(df)

    pickup_heat_pd = compute_geo_heatmap(
        df, "pickup_longitude", "pickup_latitude", PICKUP_LON_MIN, PICKUP_LON_MAX, PICKUP_LAT_MIN, PICKUP_LAT_MAX
    )
    dropoff_heat_pd = compute_geo_heatmap(
        df, "dropoff_longitude", "dropoff_latitude", DROPOFF_LON_MIN, DROPOFF_LON_MAX, DROPOFF_LAT_MIN, DROPOFF_LAT_MAX
    )
    fare_distance_heat_pd = compute_fare_distance_heatmap(df)

    save_dataframe(fare_hist_pd, FARE_HIST_CSV)
    save_dataframe(distance_hist_pd, DIST_HIST_CSV)
    save_dataframe(passenger_dist_pd, PASSENGER_CSV)
    save_dataframe(hour_dist_pd, HOUR_CSV)
    save_dataframe(monthly_trips_pd, MONTHLY_CSV)
    save_dataframe(pickup_heat_pd, PICKUP_HEAT_CSV)
    save_dataframe(dropoff_heat_pd, DROPOFF_HEAT_CSV)
    save_dataframe(fare_distance_heat_pd, FARE_DISTANCE_HEAT_CSV)

    # Correlation matrix (semua fitur yang akan dipakai modeling + target)
    # Menggunakan FEATURE_COLS dari config untuk konsistensi
    from config import FEATURE_COLS
    
    corr_columns = ["fare_amount"] + FEATURE_COLS  # Target + semua fitur
    
    # Verifikasi kolom ada di dataframe
    available_cols = [c for c in corr_columns if c in df.columns]
    if len(available_cols) < len(corr_columns):
        missing_cols = set(corr_columns) - set(available_cols)
        print(f"Warning: Kolom tidak tersedia untuk correlation: {missing_cols}")
    
    corr_matrix = pd.DataFrame(index=available_cols, columns=available_cols, dtype=float)
    for c1 in available_cols:
        for c2 in available_cols:
            try:
                corr_matrix.loc[c1, c2] = df.stat.corr(c1, c2)
            except Exception as e:
                print(f"Error calculating correlation for {c1} vs {c2}: {e}")
                corr_matrix.loc[c1, c2] = 0.0
    
    corr_matrix = corr_matrix.fillna(0.0)
    corr_matrix.to_csv(CORR_CSV)
    print(f"Correlation matrix tersimpan di {CORR_CSV}")

    # Dashboard plot
    pickup_heat_pivot = pivot_heatmap(pickup_heat_pd)
    dropoff_heat_pivot = pivot_heatmap(dropoff_heat_pd)
    fare_distance_heat_pivot = pd.DataFrame()
    if not fare_distance_heat_pd.empty:
        renamed = fare_distance_heat_pd.rename(
            columns={
                "fare_bin": "lat_bin",
                "dist_bin": "lon_bin",
                "fare_center": "lat_center",
                "dist_center": "lon_center",
            }
        )
        fare_distance_heat_pivot = pivot_heatmap(renamed)

    plot_dashboard(
        fare_hist_pd,
        distance_hist_pd,
        passenger_dist_pd,
        monthly_trips_pd,
        hour_dist_pd,
        pickup_heat_pivot,
        dropoff_heat_pivot,
        fare_distance_heat_pivot,
        corr_matrix,
    )

    # Summary text
    summary_lines = [
        f"Total rows                    : {total_rows:,}",
        f"Missing value CSV             : {MISSING_CSV}",
        f"Numeric min/max CSV           : {MIN_MAX_CSV}",
        f"Numeric avg/std CSV           : {AVG_STD_CSV}",
        f"Fare histogram (full data)    : {FARE_HIST_CSV}",
        f"Distance histogram (full data): {DIST_HIST_CSV}",
        f"Passenger distribution        : {PASSENGER_CSV}",
        f"Hour distribution             : {HOUR_CSV}",
        f"Monthly trips                 : {MONTHLY_CSV}",
        f"Pickup heatmap CSV            : {PICKUP_HEAT_CSV}",
        f"Dropoff heatmap CSV           : {DROPOFF_HEAT_CSV}",
        f"Fare-distance heatmap CSV     : {FARE_DISTANCE_HEAT_CSV}",
        f"Correlation matrix CSV        : {CORR_CSV}",
        f"Pickup out-of-bounds          : {pickup_outside:,} rows ({pickup_outside / total_rows:.2%})",
        f"Dropoff out-of-bounds         : {dropoff_outside:,} rows ({dropoff_outside / total_rows:.2%})",
        f"Fare IQR bounds               : [{fare_iqr['lower']:.2f}, {fare_iqr['upper']:.2f}] "
        f"({fare_iqr['outliers'] / total_rows:.2%} outliers)",
        f"Distance IQR bounds           : [{dist_iqr['lower']:.2f}, {dist_iqr['upper']:.2f}] "
        f"({dist_iqr['outliers'] / total_rows:.2%} outliers)",
        f"EDA dashboard PNG             : {EDA_DASHBOARD}",
        f"Missing value plot            : {MISSING_PLOT if missing_values_pd['missing_count'].sum() else 'tidak ada missing'}",
        "Gunakan env EDA_* untuk mengubah resolusi histogram & batas visualisasi.",
    ]
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Ringkasan tertulis di {SUMMARY_TXT}")
    print(f"Dashboard visual tersimpan di {EDA_DASHBOARD}")

    spark.stop()
