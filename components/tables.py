import streamlit as st
import pandas as pd

def render_greeks_table(df_display: pd.DataFrame):
    """
    Formatea y renderiza de manera profesional la tabla de la cadena de opciones
    junto con sus métricas de riesgo (Griegas).
    
    Args:
        df_display (pd.DataFrame): DataFrame que contiene las columnas de mercado 
                                y las griegas calculadas analíticamente.
    """
    if df_display.empty:
        st.warning("No hay datos disponibles para generar la tabla de griegas.")
        return

    # Clonamos el dataframe para evitar modificar el original por referencia
    df = df_display.copy()
    
    # Establecer el precio de ejercicio (Strike) como el índice de la tabla
    if 'strike' in df.columns:
        df = df.set_index('strike')
    elif 'Strike' in df.columns:
        df = df.set_index('Strike')

    # Identificar qué griegas están presentes para aplicarles el mapa de calor (gradiente)
    greeks_cols = [col for col in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'] if col in df.columns]

    # Diccionario de formateo numérico para legibilidad de mercado
    formatter = {
        'lastPrice': '${:.2f}',
        'Theoretical Price': '${:.4f}',
        'IV': '{:.2%}',
        'Delta': '{:.3f}',
        'Gamma': '{:.4f}',
        'Vega': '{:.3f}',
        'Theta': '{:.3f}',
        'Rho': '{:.3f}',
        'volume': '{:,.0f}',
        'openInterest': '{:,.0f}'
    }
    
    # Filtrar el formateador para aplicar solo a las columnas que realmente existan en el DataFrame
    active_formatter = {k: v for k, v in formatter.items() if k in df.columns}

    # Aplicar estilos avanzados de Pandas (Gradiente + Formatos de moneda/número)
    styled_df = df.style.background_gradient(
        cmap='viridis', 
        subset=greeks_cols
    ).format(active_formatter, na_rep="-")

    # Dibujar la tabla interactiva ocupando todo el ancho disponible
    st.dataframe(styled_df, width='stretch')