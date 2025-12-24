# Import Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow

print('Import Library berhasil')

def main(data_path, tracking_uri):
    # Set up mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("water_potability_ci")
    
    print(f'MLflow Tracking URI: {tracking_uri}')
    print('Set-up berhasil')

    # Membaca dataset
    df = pd.read_csv(data_path)
    print(f'Dataset shape: {df.shape}')

    # Pisahkan fitur dan Target
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Pisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standarisasi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Autolog
    mlflow.autolog()

    with mlflow.start_run(run_name="RF_CI_CD"):

        # Buat dan latih model Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        print("Akurasi:", accuracy)
        
        # Log parameter tambahan
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        # Simpan model
        mlflow.sklearn.log_model(model, "model")
        
        return accuracy




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="water_potability_preprocessing.csv")
    parser.add_argument("--tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", ""))
    
    args = parser.parse_args()
    
    # VERIFIKASI TRACKING URI
    if not args.tracking_uri:
        print("❌ ERROR: MLFLOW_TRACKING_URI tidak diberikan.")
        print("   Cara 1: Atur secret 'MLFLOW_TRACKING_URI' di GitHub Repository Settings")
        print("   Cara 2: Berikan langsung: --tracking_uri http://localhost:5001")
        exit(1)
    
    # VERIFIKASI FILE DATASET
    if not os.path.exists(args.data_path):
        print(f"❌ ERROR: File dataset '{args.data_path}' tidak ditemukan.")
        print(f"   File di direktori saat ini: {os.listdir('.')}")
        exit(1)
    
    print(f"✅ Menggunakan dataset: {args.data_path}")
    print(f"✅ Menggunakan MLflow URI: {args.tracking_uri}")
    
    try:
        accuracy = main(args.data_path, args.tracking_uri)
        print(f"✅ Model training selesai. Akurasi: {accuracy}")
    except Exception as e:
        print(f"❌ Error selama training: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
