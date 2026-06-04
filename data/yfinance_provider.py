import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

class YFinanceProvider:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.asset = yf.Ticker(ticker)

    def get_spot_price(self) -> float:
        """Obtiene el precio actual del activo subyacente."""
        try:
            history = self.asset.history(period="1d")
            if history.empty:
                raise ValueError("No se pudieron obtener datos del ticker.")
            return float(history['Close'].iloc[-1])
        except Exception as e:
            raise RuntimeError(f"Error conectando a YFinance: {e}")

    def get_option_chain(self) -> pd.DataFrame:
        """Obtiene la cadena completa de opciones unificada y limpiada."""
        expirations = self.asset.options
        if not expirations:
            return pd.DataFrame()
        
        chains = []
        for exp in expirations: # Idealmente, limitar a 2-3 vencimientos para rendimiento inicial
            opt = self.asset.option_chain(exp)
            
            calls = opt.calls
            calls['option_type'] = 'call'
            
            puts = opt.puts
            puts['option_type'] = 'put'
            
            chain = pd.concat([calls, puts], ignore_index=True)
            chain['expiration_date'] = pd.to_datetime(exp)
            
            # Calcular días al vencimiento (DTE)
            today = pd.to_datetime('today').normalize()
            chain['dte'] = (chain['expiration_date'] - today).dt.days / 365.25
            # Evitar DTE = 0 para cálculos matemáticos
            chain['dte'] = np.where(chain['dte'] <= 0, 0.001, chain['dte']) 
            
            chains.append(chain)
            
        return pd.concat(chains, ignore_index=True)
    
    def calculate_volatility_stats(self, current_iv: float) -> dict:
        """
        Calcula el IV Rank y el IV Percentile usando la distribución móvil 
        de la volatilidad histórica del activo durante el último año.
        """
        # Descargar el historial de precios del último año (252 días hábiles)
        hist = self.asset.history(period="1y")
        if hist.empty or len(hist) < 30:
            return {"rank": 50.0, "percentile": 50.0}
        
        # 1. Calcular retornos diarios logarítmicos o porcentuales
        hist['Returns'] = hist['Close'].pct_change()
        
        # 2. Calcular Volatilidad Histórica Móvil de 21 días (1 mes de trading) anualizada (sqrt(252))
        rolling_vol = hist['Returns'].rolling(window=21).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if rolling_vol.empty:
            return {"rank": 50.0, "percentile": 50.0}
        
        # 3. Calcular IV Rank
        v_min = rolling_vol.min()
        v_max = rolling_vol.max()
        
        if v_max != v_min:
            iv_rank = ((current_iv - v_min) / (v_max - v_min)) * 100
        else:
            iv_rank = 50.0
            
        # 4. Calcular IV Percentile (Días del año donde la volatilidad fue menor que la actual)
        days_below = np.sum(rolling_vol < current_iv)
        iv_percentile = (days_below / len(rolling_vol)) * 100
        
        # Acotar matemáticamente entre 0 y 100 por seguridad visual
        return {
            "rank": max(0.0, min(100.0, iv_rank)),
            "percentile": max(0.0, min(100.0, iv_percentile))
        }

    def calculate_put_call_ratio(self, df_options: pd.DataFrame) -> float:
        """
        Calcula el Put-Call Ratio de volumen ponderando la cadena completa de opciones.
        """
        if df_options.empty or 'option_type' not in df_options.columns or 'volume' not in df_options.columns:
            return 1.0
            
        # Limpiar filas sin volumen reportado
        df_clean = df_options.dropna(subset=['volume', 'option_type'])
        
        # Sumar volúmenes por tipo de contrato
        call_vol = df_clean[df_clean['option_type'].str.lower() == 'call']['volume'].sum()
        put_vol = df_clean[df_clean['option_type'].str.lower() == 'put']['volume'].sum()
        
        if call_vol == 0:
            return 0.0 if put_vol == 0 else 2.0
            
        return put_vol / call_vol
