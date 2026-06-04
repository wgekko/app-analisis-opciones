import pandas as pd
import numpy as np

def detect_price_anomalies(df_display: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    """
    Compara el precio de cotización (lastPrice) contra el precio analítico de Black-Scholes.
    Calcula el Z-Score de la desviación porcentual para identificar ineficiencias de mercado.
    """
    if df_display.empty or 'lastPrice' not in df_display.columns or 'Theoretical Price' not in df_display.columns:
        return df_display
    
    df = df_display.copy()
    
    # Filtrar opciones ilíquidas o con precio cercano a cero para evitar divisiones inválidas o ruido
    df = df[(df['lastPrice'] > 0.05) & (df['Theoretical Price'] > 0.05)]
    
    if df.empty:
        return df
        
    # 1. Calcular la desviación porcentual respecto al modelo teórico
    df['Price_Dev_Pct'] = (df['lastPrice'] - df['Theoretical Price']) / df['Theoretical Price']
    
    # 2. Calcular métricas estadísticas de la muestra
    mean_dev = df['Price_Dev_Pct'].mean()
    std_dev = df['Price_Dev_Pct'].std()
    
    # 3. Calcular Z-Score de la anomalía
    if std_dev > 0:
        df['Z_Score'] = (df['Price_Dev_Pct'] - mean_dev) / std_dev
    else:
        df['Z_Score'] = 0.0
        
    # 4. Clasificar anomalías según el umbral estadístico (Z-Score)
    df['Alerta_Sujeta'] = 'Normal'
    df.loc[df['Z_Score'] > z_threshold, 'Alerta_Sujeta'] = '🚨 Sobrevaluada (Cara)'
    df.loc[df['Z_Score'] < -z_threshold, 'Alerta_Sujeta'] = '🟢 Subvaluada (Barata)'
    
    return df