# Import Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow
import argparse

print('Import Library berhasil')

def train_model(data_path, tracking_uri):
    # Set up mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("water_potability_ci")

    print('Set-up berhasil')

    # Membaca dataset
    df = pd.read_csv(data_path)

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

    with mlflow.start_run(run_name = "RF_ci"):

        # Buat dan latih model Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        print("Akurasi:", accuracy)
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="water_potability_preprocessing.csv")
    parser.add_argument("--tracking_uri", type=str, default="http://localhost:5001")
    args = parser.parse_args()

    train_model(args.data_path, args.tracking_uri)
