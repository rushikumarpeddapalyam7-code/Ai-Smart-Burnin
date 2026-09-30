import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class BurnInAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.1, random_state=42)
        
    def fit_predict(self, df: pd.DataFrame):
        # Extract numeric parametric columns (e.g., 0h, 24h, 96h, 168h metrics)
        feature_cols = [col for col in df.columns if any(t in col for t in ['0h', '24h', '96h', '168h', 'iddq', 'leakage', 'delay'])]
        
        if not feature_cols:
            # Fallback to all numeric columns except ID
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        results_df = df.copy()
        results_df['Anomaly_Score'] = scores
        # Convert to binary status: 0 = Normal (Pass), 1 = Latent Defect Risk (Review/Fail)
        results_df['Status'] = np.where(predictions == -1, 'Review / Latent Risk', 'Pass')
        
        return results_df, feature_cols