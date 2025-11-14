# main.py
"""
NYC Taxi Fare Prediction - Pipeline Orchestrator
Menjalankan seluruh pipeline machine learning secara berurutan.

Urutan eksekusi:
1. load_data.py       - Load & clean raw data
2. eda.py             - Exploratory Data Analysis
3. preprocessing.py   - Feature engineering & filtering
4. vectorize.py       - Vectorization & train/test split
5. modelling.py       - Train baseline models (LR, RF, GBT)
6. evaluate.py        - Evaluate & compare models
7. importance.py      - Feature importance analysis
8. tuning.py          - Hyperparameter tuning (Random Forest)
9. tuning_linear.py   - Hyperparameter tuning (Linear Regression)
10. predictions.py    - Generate actual vs predicted comparison
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_script(script_name, description=""):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {script_name}")
    if description:
        print(f"  {description}")
        print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {script_name} selesai dalam {elapsed:.1f} detik")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR di {script_name} setelah {elapsed:.1f} detik")
        print(f"   Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ File {script_name} tidak ditemukan!")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline dibatalkan oleh user di {script_name}")
        sys.exit(1)

def check_prerequisites():
    """Check if required files exist"""
    print_header("CHECKING PREREQUISITES")
    
    required_files = [
        "config.py",
        "spark_utils.py",
        "load_data.py",
        "eda.py",
        "preprocessing.py",
        "vectorize.py",
        "modelling.py",
        "evaluate.py",
        "importance.py",
        "tuning.py",
        "tuning_linear.py",
        "predictions.py"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
            print(f"❌ {file} - NOT FOUND")
        else:
            print(f"✅ {file}")
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    # Check if data file exists
    data_file = os.path.join("data", "train.csv")
    if not os.path.exists(data_file):
        print(f"\n❌ Data file not found: {data_file}")
        return False
    else:
        print(f"✅ {data_file}")
    
    print("\n✅ All prerequisites OK")
    return True

def main():
    """Main pipeline orchestrator"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("  NYC TAXI FARE PREDICTION - FULL ML PIPELINE")
    print("  PySpark MLlib Implementation")
    print("="*70)
    print(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Pipeline aborted - prerequisites not met")
        sys.exit(1)
    
    # Define pipeline stages
    pipeline = [
        ("load_data.py", "Load & clean raw data dari CSV"),
        ("eda.py", "Exploratory Data Analysis - statistik & visualisasi"),
        ("preprocessing.py", "Feature engineering & data filtering"),
        ("vectorize.py", "Feature vectorization & train/test split"),
        ("modelling.py", "Train baseline models (LR, RF, GBT)"),
        ("evaluate.py", "Evaluate & compare model performance"),
        ("importance.py", "Analyze feature importance"),
        ("tuning.py", "Hyperparameter tuning - Random Forest"),
        ("tuning_linear.py", "Hyperparameter tuning - Linear Regression"),
        ("predictions.py", "Generate actual vs predicted comparison table")
    ]
    
    print(f"\n📋 Pipeline will execute {len(pipeline)} stages")
    print("   Estimated total time: 20-40 minutes\n")
    
    # Confirmation
    response = input("🚀 Ready to start? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes', '']:
        print("❌ Pipeline cancelled by user")
        sys.exit(0)
    
    # Execute pipeline
    success_count = 0
    failed_stages = []
    
    for i, (script, description) in enumerate(pipeline, 1):
        print(f"\n📍 Stage {i}/{len(pipeline)}")
        
        if run_script(script, description):
            success_count += 1
        else:
            failed_stages.append(script)
            print(f"\n⚠️  Stage {i} failed: {script}")
            
            # Ask user if they want to continue
            response = input("   Continue to next stage? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\n❌ Pipeline aborted by user")
                break
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("PIPELINE SUMMARY")
    print(f"  Start time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:      {duration}")
    print(f"  Success:       {success_count}/{len(pipeline)} stages")
    
    if failed_stages:
        print(f"  Failed stages: {', '.join(failed_stages)}")
        print("\n❌ Pipeline completed with errors")
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!")
        print("\n📊 Results saved in:")
        print("   - artifacts/eda/                  (EDA outputs)")
        print("   - artifacts/baseline_models/      (trained models)")
        print("   - artifacts/best_rf_model/        (tuned Random Forest)")
        print("   - artifacts/best_lr_model/        (tuned Linear Regression)")
        print("   - artifacts/baseline_metrics.json (evaluation metrics)")
        
        print("\n📄 Check the following reports:")
        print("   - INDEX.md                        (navigation)")
        print("   - LAPORAN_01_DATA_LOADING.md     (Section 1)")
        print("   - LAPORAN_02_EDA.md              (Section 2)")
        print("   - LAPORAN_03_PREPROCESSING.md    (Section 3)")
        print("   - LAPORAN_04_MODELING.md         (Section 4)")
        print("   - LAPORAN_05_EVALUATION.md       (Section 5)")
        print("   - LAPORAN_06_TUNING.md           (Section 6)")
        print("="*70)
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
