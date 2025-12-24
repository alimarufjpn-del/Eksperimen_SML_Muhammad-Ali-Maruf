# Import Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import json
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

print('╔══════════════════════════════════════════╗')
print('║     Water Potability Model Training      ║')
print('╚══════════════════════════════════════════╝')
print('Import Library berhasil')

# Setup environment variables
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "water_potability_ci")
mlflow_username = os.environ.get("MLFLOW_USERNAME", "")
mlflow_password = os.environ.get("MLFLOW_PASSWORD", "")

print(f'[INFO] MLflow Tracking URI: {tracking_uri}')
print(f'[INFO] Experiment Name: {experiment_name}')

# Set up mlflow dengan authentication jika ada
try:
    if mlflow_username and mlflow_password:
        # Jika menggunakan HTTP basic auth
        import urllib.parse
        parsed_url = urllib.parse.urlparse(tracking_uri)
        auth_url = f"{parsed_url.scheme}://{mlflow_username}:{mlflow_password}@{parsed_url.netloc}{parsed_url.path}"
        mlflow.set_tracking_uri(auth_url)
    else:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    print('[SUCCESS] MLflow setup completed')
except Exception as e:
    print(f'[WARNING] MLflow setup error: {e}')
    print('[INFO] Using local MLflow tracking')

# Membaca dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "water_potability_preprocessing.csv")

print(f'[INFO] Loading data from: {data_path}')
try:
    df = pd.read_csv(data_path)
    print(f'[SUCCESS] Dataset loaded. Shape: {df.shape}')
except FileNotFoundError:
    print(f'[ERROR] Dataset not found at {data_path}')
    print('[INFO] Trying alternative paths...')
    # Coba path alternatif
    data_path = "water_potability_preprocessing.csv"
    df = pd.read_csv(data_path)

# Check data quality
print(f'[INFO] Dataset columns: {df.columns.tolist()}')
print(f'[INFO] Missing values:')
print(df.isnull().sum())

# Pisahkan fitur dan Target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Balance check
print(f'\n[INFO] Target distribution:')
print(y.value_counts())
print(f'[INFO] Class ratio: {y.value_counts(normalize=True).round(3).to_dict()}')

# Pisahkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\n[INFO] Train size: {X_train.shape[0]} samples')
print(f'[INFO] Test size: {X_test.shape[0]} samples')

# Standarisasi
print('[INFO] Standardizing features...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enable MLflow autolog
mlflow.autolog()

# Start MLflow run
with mlflow.start_run(run_name=f"RF_CI_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
    run_id = run.info.run_id
    print(f'\n[INFO] MLflow Run ID: {run_id}')
    
    # Log environment info
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("platform", sys.platform)
    mlflow.log_param("ci_environment", "github_actions")
    
    # Log dataset info
    mlflow.log_param("dataset_path", data_path)
    mlflow.log_param("dataset_rows", df.shape[0])
    mlflow.log_param("dataset_features", df.shape[1])
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    
    # Log model parameters
    n_estimators = 100
    random_state = 42
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    # Buat dan latih model Random Forest
    print('[INFO] Training Random Forest model...')
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    print('[SUCCESS] Model training completed')
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f'\n[RESULTS] Model Performance:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  ROC-AUC: {roc_auc:.4f}')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    for class_label in report:
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            mlflow.log_metric(f"precision_class_{class_label}", report[class_label]['precision'])
            mlflow.log_metric(f"recall_class_{class_label}", report[class_label]['recall'])
            mlflow.log_metric(f"f1_class_{class_label}", report[class_label]['f1-score'])
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_path = os.path.join(current_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path, "feature_importance")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    importance_plot_path = os.path.join(current_dir, "feature_importance.png")
    plt.savefig(importance_plot_path)
    mlflow.log_artifact(importance_plot_path, "plots")
    plt.close()
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(current_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="WaterPotabilityClassifier"
    )
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path, "preprocessing")
    
    # Save run info
    run_info = {
        "run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    run_info_path = os.path.join(current_dir, "run_info.json")
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    mlflow.log_artifact(run_info_path, "metadata")
    
    # Print summary
    print('\n' + '='*50)
    print('TRAINING SUMMARY:')
    print('='*50)
    print(f'Run ID: {run_id}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'Features: {len(X.columns)}')
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Testing samples: {X_test.shape[0]}')
    print('='*50)
    
    # Export run_id untuk GitHub Actions
    print(f'::set-output name=run_id::{run_id}')
    
print('\n[SUCCESS] Model training pipeline completed!')
