import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class IVDirectionPredictor:
    """
    Pipeline cuántitativo de Machine Learning para predecir la dirección futura 
    de la Volatilidad Implícita (IV) promedio o de un contrato específico,
    utilizando variables macro/técnicas y rezagos del activo subyacente.
    """
    def __init__(self, model_type: str = 'random_forest', n_estimators: int = 100, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, learning_rate=0.05)
        else:
            raise ValueError("model_type soportado: 'random_forest' o 'gradient_boosting'")
            
        self.feature_cols = []
        self.is_trained = False

    def engineer_features(self, df_history: pd.DataFrame, target_horizon: int = 5) -> pd.DataFrame:
        """
        Construye las características cuantitativas (Features) basadas en el precio histórico del activo.
        
        Parámetros:
        -----------
        df_history : pd.DataFrame
            Historial del activo subyacente con columnas: 'Close', 'Volume' y opcionalmente 'Implied_Vol_History'.
        target_horizon : int
            Número de días hacia adelante para predecir el movimiento de la IV (ej. 5 días).
        """
        df = df_history.copy().sort_index()
        
        # Si no viene una columna de IV histórica simulada o real, calculamos la Volatilidad Histórica (HV) como proxy
        if 'IV_Proxy' not in df.columns:
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['IV_Proxy'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252) # HV de 20 días como proxy de la dinámica
            
        # 1. Generación de características basadas en Volatilidad e Impulso
        df['HV_10'] = df['log_ret'].rolling(window=10).std() * np.sqrt(252) if 'log_ret' in df.columns else df['IV_Proxy'].shift(1)
        df['HV_30'] = df['log_ret'].rolling(window=30).std() * np.sqrt(252) if 'log_ret' in df.columns else df['IV_Proxy'].shift(2)
        
        # Cambios porcentuales e indicadores de momento de la IV
        df['IV_pct_change_1d'] = df['IV_Proxy'].pct_change(1)
        df['IV_pct_change_5d'] = df['IV_Proxy'].pct_change(5)
        df['IV_ma_ratio'] = df['IV_Proxy'] / df['IV_Proxy'].rolling(window=20).mean()
        
        # 2. Retornos y Volúmenes del subyacente
        df['asset_ret_1d'] = df['Close'].pct_change(1)
        df['asset_ret_5d'] = df['Close'].pct_change(5)
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
        
        # 3. Definición del Target (Variable objetivo)
        # 1 si la IV sube en los próximos 'target_horizon' días, 0 si se mantiene o baja
        df['future_IV'] = df['IV_Proxy'].shift(-target_horizon)
        df['target'] = (df['future_IV'] > df['IV_Proxy']).astype(int)
        
        # Limpieza de valores nulos generados por rezagos y ventanas móviles
        df = df.dropna()
        
        self.feature_cols = [
            'HV_10', 'HV_30', 'IV_pct_change_1d', 'IV_pct_change_5d', 
            'IV_ma_ratio', 'asset_ret_1d', 'asset_ret_5d', 'volume_ma_ratio'
        ]
        
        return df

    def train(self, df_features: pd.DataFrame) -> Dict[str, float]:
        """
        Entrena el modelo predictivo y evalúa las métricas de rendimiento en test.
        """
        if df_features.empty or len(df_features) < 50:
            raise ValueError("Muestra de datos insuficiente para el entrenamiento del modelo de ML.")
            
        X = df_features[self.feature_cols]
        y = df_features['target']
        
        # División temporal o aleatoria (usamos shuffle=False para series de tiempo para evitar data leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else y_pred
        
        # Métricas
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        return metrics

    def predict_latest(self, df_features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predice la dirección de la IV para la última fila de datos disponible (tiempo real).
        Retorna: (Clase_Predicha, Probabilidad_de_Subida)
        """
        if not self.is_trained:
            raise RuntimeError("El modelo debe entrenarse antes de realizar predicciones.")
            
        latest_row = df_features[self.feature_cols].iloc[[-1]]
        prediction = int(self.model.predict(latest_row)[0])
        probability = float(self.model.predict_proba(latest_row)[0][1])
        
        return prediction, probability

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Retorna la importancia de cada característica dentro del modelo entrenado.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado aún.")
            
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, importances))