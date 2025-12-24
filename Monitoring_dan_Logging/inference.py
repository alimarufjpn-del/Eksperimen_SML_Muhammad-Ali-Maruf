"""
Script untuk inference model Water Potability
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterPotabilityModel:
    def __init__(self, model_path: str = None):
        """
        Inisialisasi model untuk inference
        
        Args:
            model_path: Path ke file model (opsional)
        """
        self.model_name = "Water Potability Classifier"
        self.version = "1.0.0"
        
        if model_path:
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using simulated model.")
                self.model = self._create_simulated_model()
        else:
            logger.info("No model provided. Using simulated model.")
            self.model = self._create_simulated_model()
    
    def _create_simulated_model(self):
        """Membuat model simulasi untuk testing"""
        class SimulatedModel:
            def __init__(self):
                self.feature_names = [
                    'ph', 'Hardness', 'Solids', 'Chloramines',
                    'Sulfate', 'Conductivity', 'Organic_carbon',
                    'Trihalomethanes', 'Turbidity'
                ]
            
            def predict(self, X):
                """Prediksi simulasi"""
                n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                
                # Prediksi random dengan bias tertentu
                predictions = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
                
                return predictions
            
            def predict_proba(self, X):
                """Probabilitas prediksi simulasi"""
                n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                
                # Probabilitas random
                probas = np.random.rand(n_samples, 2)
                probas = probas / probas.sum(axis=1, keepdims=True)
                
                return probas
        
        return SimulatedModel()
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Melakukan prediksi untuk single sample
        
        Args:
            features: Dictionary berisi fitur-fitur
            
        Returns:
            Dictionary berisi hasil prediksi
        """
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Ensure correct column order
            expected_features = [
                'ph', 'Hardness', 'Solids', 'Chloramines',
                'Sulfate', 'Conductivity', 'Organic_carbon',
                'Trihalomethanes', 'Turbidity'
            ]
            
            # Reorder columns if needed
            features_df = features_df.reindex(columns=expected_features)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            return {
                'prediction': int(prediction),
                'probability_0': float(probabilities[0]),
                'probability_1': float(probabilities[1]),
                'confidence': float(max(probabilities)),
                'model': self.model_name,
                'version': self.version
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Melakukan prediksi untuk batch samples
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Dictionary berisi hasil batch prediction
        """
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Ensure correct column order
            expected_features = [
                'ph', 'Hardness', 'Solids', 'Chloramines',
                'Sulfate', 'Conductivity', 'Organic_carbon',
                'Trihalomethanes', 'Turbidity'
            ]
            
            features_df = features_df.reindex(columns=expected_features)
            
            # Make predictions
            predictions = self.model.predict(features_df)
            probabilities = self.model.predict_proba(features_df)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'sample_id': i,
                    'prediction': int(pred),
                    'probability_0': float(prob[0]),
                    'probability_1': float(prob[1]),
                    'confidence': float(max(prob))
                })
            
            return {
                'results': results,
                'total_samples': len(results),
                'positive_count': int(sum(predictions)),
                'negative_count': int(len(predictions) - sum(predictions)),
                'model': self.model_name,
                'version': self.version
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Mendapatkan informasi model"""
        return {
            'name': self.model_name,
            'version': self.version,
            'features': [
                'ph', 'Hardness', 'Solids', 'Chloramines',
                'Sulfate', 'Conductivity', 'Organic_carbon',
                'Trihalomethanes', 'Turbidity'
            ],
            'description': 'Random Forest classifier for water potability prediction',
            'output': {
                '0': 'Not Potable',
                '1': 'Potable'
            }
        }

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi model
    model = WaterPotabilityModel()
    
    # Contoh single prediction
    sample_data = {
        'ph': 7.08,
        'Hardness': 196.84,
        'Solids': 22018.42,
        'Chloramines': 7.30,
        'Sulfate': 333.78,
        'Conductivity': 429.34,
        'Organic_carbon': 18.44,
        'Trihalomethanes': 85.18,
        'Turbidity': 4.10
    }
    
    result = model.predict_single(sample_data)
    print("Single Prediction Result:")
    print(result)
    
    # Contoh batch prediction
    batch_data = [
        {
            'ph': 7.08,
            'Hardness': 196.84,
            'Solids': 22018.42,
            'Chloramines': 7.30,
            'Sulfate': 333.78,
            'Conductivity': 429.34,
            'Organic_carbon': 18.44,
            'Trihalomethanes': 85.18,
            'Turbidity': 4.10
        },
        {
            'ph': 6.85,
            'Hardness': 210.25,
            'Solids': 19567.32,
            'Chloramines': 6.95,
            'Sulfate': 310.45,
            'Conductivity': 415.67,
            'Organic_carbon': 17.89,
            'Trihalomethanes': 82.34,
            'Turbidity': 3.95
        }
    ]
    
    batch_result = model.predict_batch(batch_data)
    print("\nBatch Prediction Result:")
    print(batch_result)
    
    # Model info
    info = model.get_model_info()
    print("\nModel Information:")
    print(info)