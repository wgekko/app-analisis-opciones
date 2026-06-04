import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import logging

logger = logging.getLogger(__name__)

def plot_volatility_surface_3d(df_options: pd.DataFrame, spot_price: float) -> go.Figure:
    """
    Genera un gráfico interactivo 3D de la Superficie de Volatilidad Implícita.
    Utiliza interpolación cúbica para recrear una superficie continua a partir de datos discretos.
    
    Parámetros:
    -----------
    df_options : pd.DataFrame
        Cadena de opciones con columnas: 'strike', 'dte' (días al vencimiento en años), 'impliedVolatility'.
    spot_price : float
        Precio actual del activo subyacente (para calcular el Moneyness).
        
    Retorna:
    --------
    plotly.graph_objects.Figure
    """
    # 1. Limpieza y filtrado de datos anómalos
    df_clean = df_options[
        (df_options['impliedVolatility'] > 0.01) & 
        (df_options['impliedVolatility'] < 2.5) & 
        (df_options['dte'] > 0)
    ].copy()
    
    if df_clean.empty or len(df_clean) < 10:
        logger.warning("Datos insuficientes para generar la superficie de volatilidad.")
        fig = go.Figure()
        fig.add_annotation(text="Datos de volatilidad insuficientes para este activo", showarrow=False, font=dict(size=16))
        fig.update_layout(template="plotly_dark")
        return fig

    # Calcular Moneyness relativo (Strike / Spot) para estandarizar el eje X
    df_clean['moneyness'] = df_clean['strike'] / spot_price
    
    # Extraer coordenadas de mercado discretas
    x_market = df_clean['moneyness'].values
    y_market = df_clean['dte'].values * 365.25  # Convertir de años a días para mejor visualización
    z_market = df_clean['impliedVolatility'].values * 100  # Convertir a porcentaje (%)

    # 2. Crear una malla (grid) regular para la interpolación lineal/cúbica
    x_grid = np.linspace(x_market.min(), x_market.max(), 50)
    y_grid = np.linspace(y_market.min(), y_market.max(), 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    # 3. Interpolar los datos discretos sobre la malla regular
    # Usamos 'linear' como fallback si 'cubic' falla por geometría de los datos
    try:
        Z = griddata((x_market, y_market), z_market, (X, Y), method='cubic')
        # Rellenar NaNs producidos en los bordes de la interpolación cúbica con interpolación lineal
        nan_mask = np.isnan(Z)
        if nan_mask.any():
            Z_linear = griddata((x_market, y_market), z_market, (X, Y), method='linear')
            Z[nan_mask] = Z_linear[nan_mask]
    except Exception as e:
        logger.error(f"Fallo en interpolación cúbica, usando lineal: {e}")
        Z = griddata((x_market, y_market), z_market, (X, Y), method='linear')

    # Rellenar remanentes con el vecino más cercano para evitar huecos en el gráfico
    if np.isnan(Z).any():
        Z_nearest = griddata((x_market, y_market), z_market, (X, Y), method='nearest')
        Z[np.isnan(Z)] = Z_nearest[np.isnan(Z)]

    # 4. Construcción del gráfico 3D con Plotly
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(title='IV (%)', titleside='top'),
            hovertemplate='Moneyness: %{x:.2f}<br>Días al Vencimiento: %{y:.1f}<br>IV: %{z:.2f}%<extra></extra>'
        )
    ])

    # Añadir los puntos reales del mercado encima de la superficie para validación visual
    fig.add_trace(go.Scatter3d(
        x=x_market, y=y_market, z=z_market,
        mode='markers',
        marker=dict(size=2, color='white', opacity=0.6),
        name='Puntos de Mercado',
        hovertemplate='Moneyness: %{x:.2f}<br>Días: %{y:.0f}<br>IV Real: %{z:.2f}%<extra></extra>'
    ))

    # Configuración de diseño profesional estilo terminal financiera
    fig.update_layout(
        title=dict(
            text="Superficie de Volatilidad Implícita 3D (Interpolada)",
            x=0.5, y=0.95,
            font=dict(size=18, color='#f5f5f5')
        ),
        scene=dict(
            xaxis=dict(title='Moneyness (Strike/Spot)', gridcolor='#334155', color='#cbd5e1'),
            yaxis=dict(title='Tiempo al Vencimiento (Días)', gridcolor='#334155', color='#cbd5e1'),
            zaxis=dict(title='Volatilidad Implícita (%)', gridcolor='#334155', color='#cbd5e1'),
            aspectratio=dict(x=1, y=1, z=0.7),
            bgcolor='#0f172a'
        ),
        paper_bgcolor='#0f172a',
        #margin=dict(l=10, right=10, b=10, t=50),
        margin=dict(l=10, r=10, b=10, t=50),
        template="plotly_dark",
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

#---------------------- adicionando los charts de la carpeta strategies --------------------

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

def plot_payoff_profile(s_t: np.ndarray, total_pnl: np.ndarray, spot_price: float):
    """
    Genera un gráfico analítico interactivo en 2D de las zonas de Ganancia/Pérdida.
    """
    fig = go.Figure()

    # Trazar la curva total de pérdidas y ganancias de la estrategia
    fig.add_trace(go.Scatter(
        x=s_t, y=total_pnl,
        mode='lines',
        name='P&L Combinado',
        line=dict(color='#a3e635', width=3.5),
        fill='tozeroy',  # Crea un sombreado estético hacia la línea cero
        fillcolor='rgba(163, 230, 53, 0.05)'
    ))

    # Línea horizontal de referencia de punto de equilibrio (P&L = 0)
    fig.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1.5)
    
    # Línea vertical indicadora de la cotización Spot actual del mercado
    fig.add_vline(
        x=spot_price, 
        line_dash="dash", 
        line_color="#38bdf8", 
        line_width=2,
        annotation_text=f"Spot Actual: ${spot_price:.2f}", 
        annotation_position="top left"
    )

    fig.update_layout(
        title="Perfil de Riesgo y Cobertura (Payoff Combinado al Vencimiento)",
        xaxis_title="Precio del Activo Subyacente al Vencimiento (S_T)",
        yaxis_title="Ganancia / Pérdida Neta ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig