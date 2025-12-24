# Import Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow

print('Import Library berhasil')

# Set up mlflow
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("water_potability_local")

print('Set-up berhasil')

# Membaca dataset
df = pd.read_csv("/Users/zero/Project/Membangun_model/water_potability_preprocessing.csv")

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

with mlflow.start_run(run_name = "RF_local"):

    # Buat dan latih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    print("Akurasi:", accuracy)