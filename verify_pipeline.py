"""
Verification script to test all ML Pipeline components
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        from src.data_processing import load_data, aggregate_customer_features, preprocessing_pipeline
        print("✅ data_processing.py - All functions imported successfully")
        print("   - load_data()")
        print("   - aggregate_customer_features()")
        print("   - preprocessing_pipeline()")
    except ImportError as e:
        print(f"❌ data_processing.py - Import failed: {e}")
        return False
    
    try:
        from src.rfm import calculate_rfm, create_proxy_target
        print("\n✅ rfm.py - All functions imported successfully")
        print("   - calculate_rfm()")
        print("   - create_proxy_target()")
    except ImportError as e:
        print(f"❌ rfm.py - Import failed: {e}")
        return False
    
    try:
        from src.train import run_training_pipeline
        print("\n✅ train.py - Training function imported successfully")
        print("   - run_training_pipeline()")
    except ImportError as e:
        print(f"❌ train.py - Import failed: {e}")
        return False
    
    try:
        import mlflow
        print(f"\n✅ MLflow imported successfully (v{mlflow.__version__})")
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")
        return False
    
    return True

def test_sklearn_components():
    """Test sklearn components"""
    print("\n" + "=" * 60)
    print("TESTING SKLEARN COMPONENTS")
    print("=" * 60)
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        
        print("✅ All required sklearn components available:")
        print("   - Pipeline")
        print("   - StandardScaler")
        print("   - KMeans")
        print("   - LogisticRegression")
        print("   - RandomForestClassifier")
        print("   - train_test_split")
        print("   - roc_auc_score")
        return True
    except ImportError as e:
        print(f"❌ sklearn components import failed: {e}")
        return False

def check_artifacts():
    """Check if model artifacts exist"""
    print("\n" + "=" * 60)
    print("CHECKING SAVED ARTIFACTS")
    print("=" * 60)
    
    models_dir = "models"
    expected_files = [
        "logistic_regression_model.pkl",
        "random_forest_model.pkl",
        "preprocessor.pkl"
    ]
    
    if not os.path.exists(models_dir):
        print(f"⚠️  Models directory '{models_dir}/' not found")
        print("   Run 'python -m src.train' to create models")
        return False
    
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"✅ {filename} ({size_kb:.2f} KB)")
        else:
            print(f"❌ {filename} - NOT FOUND")
            all_exist = False
    
    # Check MLflow directory
    if os.path.exists("mlruns"):
        print("\n✅ MLflow tracking directory exists (mlruns/)")
    else:
        print("\n⚠️  MLflow tracking directory not found")
        print("   Run 'python -m src.train' to create MLflow runs")
    
    return all_exist

def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from src.config import config
        print("✅ Configuration loaded successfully")
        print(f"\n   Model Configuration:")
        print(f"   - random_state: {config.model.random_state}")
        print(f"   - test_size: {config.model.test_size}")
        print(f"   - n_clusters_risk: {config.model.n_clusters_risk}")
        print(f"   - rf_n_estimators: {config.model.rf_n_estimators}")
        print(f"   - rf_max_depth: {config.model.rf_max_depth}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("\n" + "🔍 " * 20)
    print("ML PIPELINE VERIFICATION SCRIPT")
    print("🔍 " * 20 + "\n")
    
    results = {
        "Imports": test_imports(),
        "Sklearn Components": test_sklearn_components(),
        "Configuration": test_config(),
        "Artifacts": check_artifacts()
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - ML PIPELINE IS READY!")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ABOVE")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("   1. Run training: python -m src.train")
    print("   2. View MLflow UI: python -m mlflow ui")
    print("   3. Access UI at: http://localhost:5000")
    print()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
