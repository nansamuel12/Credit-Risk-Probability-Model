import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import os

from src.data_processing import load_data, aggregate_customer_features, preprocessing_pipeline
from src.rfm import calculate_rfm, create_proxy_target
from src.config import config

def run_training_pipeline():
    """Main training pipeline with MLflow integration."""
    
    # Set MLflow experiment
    mlflow.set_experiment("Credit-Risk-Probability-Model")
    
    with mlflow.start_run(run_name="credit_risk_training"):
        # 1. Load Data
        print("--- Loading Raw Data ---")
        raw_path = str(config.paths.raw_data)
        df = load_data(raw_path)
        
        # Log data stats
        mlflow.log_param("raw_data_rows", len(df))
        mlflow.log_param("raw_data_cols", df.shape[1])
        
        # 2. Engineering RFM and Proxy Target
        print("--- Engineering RFM Features & Proxy Target ---")
        df_rfm = calculate_rfm(df)
        df_labeled = create_proxy_target(df_rfm, random_state=config.model.random_state)
        
        # Log RFM stats
        mlflow.log_param("n_customers", len(df_labeled))
        mlflow.log_param("n_clusters", config.model.n_clusters_risk)
        
        # 3. Prepare Features and Target
        X = df_labeled[['Recency', 'Frequency', 'Monetary']]
        y = df_labeled['Risk_Label']
        
        # Log class distribution
        class_dist = y.value_counts()
        mlflow.log_metric("class_0_count", int(class_dist[0]))
        mlflow.log_metric("class_1_count", int(class_dist[1]))
        mlflow.log_metric("class_imbalance_ratio", float(class_dist[1] / class_dist[0]))
        
        # 4. Train/Test Split
        print("--- Splitting Data ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.model.test_size, 
            random_state=config.model.random_state, 
            stratify=y
        )
        
        # Log split parameters
        mlflow.log_param("test_size", config.model.test_size)
        mlflow.log_param("random_state", config.model.random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # 5. Preprocessing (Scaling numeric cols)
        print("--- Preprocessing ---")
        preprocessor = preprocessing_pipeline(num_cols=['Recency', 'Frequency', 'Monetary'])
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # 6. Train Models
        print("--- Training Models ---")
        
        # ========== Logistic Regression ==========
        print("\n[1/2] Training Logistic Regression...")
        lr = LogisticRegression(random_state=config.model.random_state, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
        lr_preds = lr.predict(X_test_scaled)
        lr_auc = roc_auc_score(y_test, lr_probs)
        
        # Log LR parameters and metrics
        mlflow.log_param("lr_max_iter", 1000)
        mlflow.log_metric("lr_roc_auc", lr_auc)
        
        print(f"✓ Logistic Regression ROC-AUC: {lr_auc:.4f}")
        
        # ========== Random Forest ==========
        print("\n[2/2] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=config.model.rf_n_estimators,
            max_depth=config.model.rf_max_depth,
            random_state=config.model.random_state,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
        rf_preds = rf.predict(X_test_scaled)
        rf_auc = roc_auc_score(y_test, rf_probs)
        
        # Log RF parameters and metrics
        mlflow.log_param("rf_n_estimators", config.model.rf_n_estimators)
        mlflow.log_param("rf_max_depth", config.model.rf_max_depth)
        mlflow.log_metric("rf_roc_auc", rf_auc)
        
        print(f"✓ Random Forest ROC-AUC: {rf_auc:.4f}")
        
        # 7. Print Performance Summary
        print("\n" + "="*60)
        print("PERFORMANCE METRICS (ROC-AUC)")
        print("="*60)
        print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
        print(f"Random Forest ROC-AUC:       {rf_auc:.4f}")
        print("="*60)
        
        # Determine best model
        if rf_auc > lr_auc:
            best_model = rf
            best_model_name = "RandomForest"
            best_auc = rf_auc
            mlflow.log_param("best_model", "RandomForest")
        else:
            best_model = lr
            best_model_name = "LogisticRegression"
            best_auc = lr_auc
            mlflow.log_param("best_model", "LogisticRegression")
        
        mlflow.log_metric("best_model_roc_auc", best_auc)
        
        print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
        
        # 8. Save Models and Preprocessor
        print("\n--- Saving Artifacts ---")
        models_dir = str(config.paths.models_dir)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save with joblib
        lr_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        joblib.dump(lr, lr_path)
        joblib.dump(rf, rf_path)
        joblib.dump(preprocessor, preprocessor_path)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(lr_path, artifact_path="models")
        mlflow.log_artifact(rf_path, artifact_path="models")
        mlflow.log_artifact(preprocessor_path, artifact_path="models")
        
        # 9. Model Registry Management
        print("\n--- Registering Models to MLflow Model Registry ---")
        
        # Register best model
        best_model_registry_name = f"credit_risk_{best_model_name.lower()}"
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        
        registered_model_version = mlflow.register_model(
            model_uri=model_uri,
            name=best_model_registry_name
        )
        
        print(f"✓ Registered model: {best_model_registry_name}")
        print(f"✓ Version: {registered_model_version.version}")
        
        # Register preprocessor
        preprocessor_registry_name = "credit_risk_preprocessor"
        preprocessor_uri = f"runs:/{mlflow.active_run().info.run_id}/preprocessor"
        
        registered_preprocessor_version = mlflow.register_model(
            model_uri=preprocessor_uri,
            name=preprocessor_registry_name
        )
        
        print(f"✓ Registered preprocessor: {preprocessor_registry_name}")
        print(f"✓ Version: {registered_preprocessor_version.version}")
        
        # 10. Promote Best Model to Production
        print("\n--- Promoting Best Model to Production ---")
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Transition best model to Production stage
        client.transition_model_version_stage(
            name=best_model_registry_name,
            version=registered_model_version.version,
            stage="Production",
            archive_existing_versions=True  # Archive previous production versions
        )
        
        # Transition preprocessor to Production stage
        client.transition_model_version_stage(
            name=preprocessor_registry_name,
            version=registered_preprocessor_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✓ {best_model_registry_name} v{registered_model_version.version} → Production")
        print(f"✓ {preprocessor_registry_name} v{registered_preprocessor_version.version} → Production")
        print(f"✓ Previous versions archived")
        
        # Add model description/tags
        client.update_model_version(
            name=best_model_registry_name,
            version=registered_model_version.version,
            description=f"Best performing model ({best_model_name}) with ROC-AUC: {best_auc:.4f}"
        )
        
        client.set_model_version_tag(
            name=best_model_registry_name,
            version=registered_model_version.version,
            key="auc_score",
            value=str(round(best_auc, 4))
        )
        
        client.set_model_version_tag(
            name=best_model_registry_name,
            version=registered_model_version.version,
            key="model_type",
            value=best_model_name
        )
        
        print(f"✓ Model tags and description updated")
        
        # 11. Log feature importance (if Random Forest is best)
        if best_model_name == "RandomForest":
            feature_names = ['Recency', 'Frequency', 'Monetary']
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            for feat, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feat}", importance)
            
            print("\n--- Feature Importance (Random Forest) ---")
            for feat, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"{feat:15s}: {importance:.4f}")
        
        print("\n" + "="*60)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Best Model: {best_model_registry_name} v{registered_model_version.version}")
        print(f"Stage: Production")
        print(f"ROC-AUC: {best_auc:.4f}")
        print("="*60)
        
        return {
            'lr_model': lr,
            'rf_model': rf,
            'preprocessor': preprocessor,
            'lr_auc': lr_auc,
            'rf_auc': rf_auc,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'model_version': registered_model_version.version,
            'run_id': mlflow.active_run().info.run_id
        }

if __name__ == "__main__":
    results = run_training_pipeline()
