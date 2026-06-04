import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class IVValueRegressor:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.features = ['HV_10', 'HV_21', 'Parkinson_Proxy', 'SMA_50', 'Volume_Ratio']
        
    def engineer_features(self, df_hist: pd.DataFrame, target_horizon: int = 5) -> pd.DataFrame:
        """Genera variables cuantitativas predictoras de volatilidad a partir de precios."""
        df = df_hist.copy()
        df['Returns'] = df['Close'].pct_change()
        
        # Volatilidades históricas rodantes anualizadas
        df['HV_10'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)
        df['HV_21'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Volatilidad de Parkinson (Captura la oscilación intradía High vs Low)
        df['Parkinson_Proxy'] = np.sqrt(252 * (np.log(df['High'] / df['Low'])**2) / (4 * np.log(2)))
        
        # Características de tendencia de precio y anomalías de volumen diario
        df['SMA_50'] = df['Close'] / df['Close'].rolling(window=50).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=21).mean()
        
        # Target: El valor exacto del régimen de volatilidad en T + horizonte (5 días)
        df['Target_IV'] = df['Parkinson_Proxy'].shift(-target_horizon)
        
        return df.dropna()
        
    def train(self, df_features: pd.DataFrame):
        """Entrena el regresor de forma cronológica (apto para series de tiempo)."""
        X = df_features[self.features]
        y = df_features['Target_IV']
        
        # Separación temporal estricta (no aleatoria) para no filtrar datos del futuro al pasado
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }
        
    def predict_latest(self, df_features: pd.DataFrame) -> float:
        """Genera la predicción numérica para los siguientes 5 días."""
        latest_x = df_features[self.features].iloc[[-1]]
        return float(self.model.predict(latest_x)[0])
        
    def get_feature_importances(self):
        return dict(zip(self.features, self.model.feature_importances_))