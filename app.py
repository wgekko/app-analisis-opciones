import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go
from data.yfinance_provider import YFinanceProvider
from pricing.black_scholes import BlackScholesModel
from components.charts import plot_volatility_surface_3d
from ml_models.iv_predictor import IVDirectionPredictor
from pricing.montecarlo import QuantumMonteCarloPricer
from components.tables import render_greeks_table
from strategies.builder import get_strategy_presets
from strategies.payoff import calculate_strategy_payoff
from components.charts import plot_payoff_profile
from pricing.classic_montecarlo import classic_monte_carlo_price
from pricing.binomial import binomial_tree_price
from ml_models.iv_regressor import IVValueRegressor
from ml_models.anomaly_detector import detect_price_anomalies
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Quant Options Analytics", layout="wide", page_icon=":material/analytics:")

col_izq, col_central, col_der = st.columns([1, 10, 1])

with col_central:    
    st.subheader(":material/terminal: Terminal Quant Avanzada Opciones/Futuros", anchor=False, text_alignment="center")
    # 2. CSS para centrar el texto dentro de los componentes st.info (o alertas)
    st.markdown("""
        <style>
        .stAlert > div {
            text-align: center;
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)
    # (Aquí continuarían tus animaciones y botones...)
    v1, v2 = st.columns(2)

# --- Carga de Animaciones ---
def load_html(file_name):
    return Path(file_name).read_text(encoding="utf-8")

try:
    opcion1_html = load_html("static/index2.html") #
    opcion2_html = load_html("static/texto-2.html") #
except:
    opcion1_html = opcion2_html = ""

# --- Función para cargar HTML como Data URL para st.iframe ---
def get_html_data_url(file_path):
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        b64 = base64.b64encode(content.encode()).decode()
        return f"data:text/html;base64,{b64}"
    except:
        return ""

# Obtener las URLs de datos
opcion1_html = get_html_data_url("static/index2.html")
opcion2_html = get_html_data_url("static/texto-2.html")

# --- RENDERIZADO ---
col_izq, col_central, col_der = st.columns([1, 10, 1]) #
with col_central:
    # 1. Animaciones en ventanas paralelas
    v1, v2 = st.columns(2)
    with v1:        
        #components.html(tunnel_html, height=400, scrolling=False)
        if opcion1_html:
            st.iframe(opcion1_html, height=320)
    with v2:
        #components.html(crt_html, height=400, scrolling=False)
        if opcion2_html:
            st.iframe(opcion2_html, height=320)
    st.write("") 

st.markdown("---")

with st.expander(":material/terminal: Desplegar la terminal con análisis de opciones"):

    # Aplicar caché a los datos de mercado
    @st.cache_data(ttl=300) # Cache por 5 minutos
    def load_market_data(ticker: str):
        provider = YFinanceProvider(ticker)
        spot = provider.get_spot_price()
        chain = provider.get_option_chain()
        return spot, chain

    # --- SIDEBAR: Navegación y Controles ---
    st.sidebar.subheader(":material/terminal: Terminal Cuantitativa")
    modulo = st.sidebar.radio("Módulos", [
        "Dashboard General", 
        "Option Chain Analyzer",
        "Greeks & Pricing",
        "Estrategias Multi-Leg",  # <-- Nueva opción agregada aquí
        "Machine Learning Predictivo",
        "Quantum Finance"
    ])
    ticker_input = st.sidebar.text_input("Ticker del Activo", value="SPY").upper()
    risk_free = st.sidebar.number_input("Risk Free Rate (%)", value=5.0) / 100

    try:
        with st.spinner(f"Cargando datos para {ticker_input}..."):
            spot_price, df_options = load_market_data(ticker_input)
        
        # --- MÓDULO 1: DASHBOARD ---
    # --- MÓDULO 1: DASHBOARD ---
        if modulo == "Dashboard General":
            st.subheader(f"Dashboard General: {ticker_input}")
            
            if df_options.empty: 
                st.warning(":material/warning: No hay datos disponibles para desplegar el Dashboard.")
            else:
                # Calcular la IV promedio actual de la cadena
                avg_iv_decimal = df_options['impliedVolatility'].mean()
                avg_iv_pct = avg_iv_decimal * 100
                
                # Instanciar proveedor y calcular métricas analíticas
                provider = YFinanceProvider(ticker_input)
                vol_stats = provider.calculate_volatility_stats(avg_iv_decimal)
                pcr_ratio = provider.calculate_put_call_ratio(df_options)
                
                # --- FILA 1: Métricas de Mercado Core ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Spot Actual", f"${spot_price:.2f}")
                col2.metric("Volatilidad Implícita (IV) Promedio", f"{avg_iv_pct:.2f}%")
                col3.metric("Opciones Activas en Cadena", f"{len(df_options):,}")
                
                st.divider()
                st.subheader("Estadísticas de Volatilidad Avanzada & Sentimiento")
                
                # --- FILA 2: Métricas Cuantitativas de Contexto ---
                g1, g2, g3 = st.columns(3)
                
                # IV Rank Card
                g1.metric(
                    label="IV Rank (1 Año)", 
                    value=f"{vol_stats['rank']:.1f}%",
                    help="Mide la IV actual respecto al mínimo y máximo del último año. Valores > 50% indican primas caras ideales para estrategias de venta (crédito)."
                )
                
                # IV Percentile Card
                g2.metric(
                    label="IV Percentile (1 Año)", 
                    value=f"{vol_stats['percentile']:.1f}%",
                    help="Porcentaje de días del año donde la volatilidad estuvo por debajo del nivel actual. Si es muy alto, se espera una reversión a la media."
                )
                
                # Put-Call Ratio Card con Delta Dinámico de color
                sentiment_text = "Sentimiento: BAJISTA (Cobertura)" if pcr_ratio > 1.0 else "Sentimiento: ALCISTA (Optimismo)"
                sentiment_color = "inverse" if pcr_ratio > 1.0 else "normal" # Rojo si es bajista, verde si es alcista
                
                g3.metric(
                    label="Put-Call Ratio (Volumen)", 
                    value=f"{pcr_ratio:.2f}",
                    delta=sentiment_text,
                    delta_color=sentiment_color,
                    help="Relación de volumen Put/Call. Un ratio mayor a 1 significa que se están negociando más contratos defensivos o bajistas en el mercado."
                )
                
                # Gráfico del Histograma (Mantener el que ya tenías abajo)
                st.markdown("### Histograma de Volatilidad Implícita")
                fig = go.Figure(data=[go.Histogram(x=df_options['impliedVolatility']*100, nbinsx=50, marker_color='#38bdf8')])
                fig.update_layout(xaxis_title="IV (%)", yaxis_title="Frecuencia de Contratos", template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, width='stretch')
                
        # --- MÓDULO 2: OPTION CHAIN ANALYZER ---
        elif modulo == "Option Chain Analyzer":
            st.subheader(f"Option Chain & Greeks Analyzer: {ticker_input}")
            
            if df_options.empty:
                st.warning(":material/warning: No hay datos de opciones disponibles para este ticker.")
            else:
                col_f1, col_f2 = st.columns(2)
                fechas = sorted(df_options['expiration_date'].dt.strftime('%Y-%m-%d').unique())
                selected_date = col_f1.selectbox("Fecha de Expiración", fechas)
                option_type = col_f2.selectbox("Tipo de Opción", ["CALL", "PUT"])
                
                df_filtered = df_options[(df_options['expiration_date'].dt.strftime('%Y-%m-%d') == selected_date) & 
                                        (df_options['option_type'] == option_type.lower())].copy()
                
                greeks_data = []
                for _, row in df_filtered.iterrows():
                    bsm = BlackScholesModel(
                        S=spot_price, K=row['strike'], T=row['dte'], 
                        r=risk_free, sigma=row['impliedVolatility']
                    )
                    greeks = bsm.greeks(option_type=option_type.lower())
                    greeks['Theoretical Price'] = bsm.price(option_type=option_type.lower())
                    greeks['Strike'] = row['strike']
                    greeks['IV'] = row['impliedVolatility']
                    greeks_data.append(greeks)
                    
                df_greeks = pd.DataFrame(greeks_data)
                df_display = pd.merge(df_filtered[['strike', 'lastPrice', 'volume', 'openInterest']], 
                                    df_greeks, left_on='strike', right_on='Strike')
                st.divider()
                st.info("tabla de datos básico")
                #st.dataframe(df_display.set_index('strike').style.background_gradient(cmap='viridis', subset=['Delta', 'Gamma', 'Vega']))            
                st.dataframe(df_display.set_index('strike').style.background_gradient(cmap='viridis', subset=['Delta', 'Gamma', 'Vega', 'Vanna', 'Volga']))
                st.divider()
                st.info("tabla de datos destacados")
                render_greeks_table(df_display)
                st.divider()

            fig_surface = plot_volatility_surface_3d(df_options, spot_price)
            st.plotly_chart(fig_surface, width='stretch') 

        # --- MÓDULO 3: CALCULADORA INDIVIDUAL DE GRIEGAS ---
        elif modulo == "Greeks & Pricing":
            st.subheader(f"Calculadora Avanzada de Griegas: {ticker_input}")
            st.markdown("Evalúa un contrato de opción individual y calcula analíticamente sus sensibilidades marginales de riesgo.")
            
            col_p1, col_p2 = st.columns([1, 2])
            
            with col_p1:
                st.subheader("Parámetros del Contrato")
                opt_type = st.selectbox("Tipo de Contrato", ["Call", "Put"])
                strike_p = st.number_input("Precio de Ejercicio (K)", value=round(spot_price, 2))
                dte_p = st.number_input("Días al Vencimiento (DTE)", value=30.0, min_value=1.0) / 365.25
                div_yield = st.number_input("Dividend Yield (q %)", value=0.0, min_value=0.0, max_value=20.0) / 100
                
                default_vol = df_options['impliedVolatility'].mean() if not df_options.empty else 0.20
                vol_p = st.slider("Volatilidad Implícita (IV %)", min_value=5, max_value=150, value=int(default_vol*100)) / 100
                
            with col_p2:
                st.subheader("Análisis Matemático (Black-Scholes OO)")
                bs_model = BlackScholesModel(S=spot_price, K=strike_p, T=dte_p, r=risk_free, sigma=vol_p, q=div_yield)
                
                bs_price = bs_model.price(opt_type.lower())
                bs_greeks = bs_model.greeks(opt_type.lower())
                
                st.metric(label=f"Precio Teórico de la {opt_type}", value=f"${bs_price:.4f}")
                st.divider()     
                st.subheader("Las Métricas Griegas del Contrato")
                g1, g2, g3, g4, g5, g6, g7 = st.columns(7)
                g1.metric("Delta (Δ)", f"{bs_greeks['Delta']:.3f}", help="Dirección del subyacente")
                g2.metric("Gamma (Γ)", f"{bs_greeks['Gamma']:.4f}", help="Aceleración de Delta")
                g3.metric("Vega (V)", f"{bs_greeks['Vega']:.3f}", help="Sensibilidad a la IV (+1%)")
                g4.metric("Theta (Θ)", f"{bs_greeks['Theta']:.3f}", help="Decaimiento temporal diario")
                g5.metric("Rho (ρ)", f"{bs_greeks['Rho']:.3f}", help="Sensibilidad a tasas de interés")
                g6.metric("Vanna", f"{bs_greeks['Vanna']:.4f}", help="Cambio de Delta ante movimientos de Volatilidad")
                g7.metric("Volga", f"{bs_greeks['Volga']:.4f}", help="Aceleración de la Vega (Convexidad de IV)")

        # --- NUEVO MÓDULO: ESTRATEGIAS MULTI-LEG ---
        if modulo == "Estrategias Multi-Leg":
            st.subheader("Constructor de Estrategias Combinadas (Multi-Leg)")
            st.markdown("Estructura estructuras complejas (Spreads, Condors o Straddles) y evalúa su perfil asimétrico de riesgo.")
            
            # Selección de Plantillas Estructurales
            presets = get_strategy_presets(spot_price)
            selected_preset = st.selectbox("Seleccionar Plantilla de Estrategia", list(presets.keys()))
            
            st.markdown("### Configuración de las Patas (Legs)")
            st.caption(":material/settings: Puedes editar directamente los valores en la tabla, añadir filas al final o eliminarlas seleccionándolas.")
            
            # Cargar los datos iniciales según la plantilla seleccionada
            initial_data = presets[selected_preset]
            
            # Crear la tabla interactiva editable
            edited_legs = st.data_editor(
                initial_data,
                num_rows="dynamic",
                column_config={
                    "action": st.column_config.SelectboxColumn("Acción", options=["buy", "sell"], required=True),
                    "type": st.column_config.SelectboxColumn("Tipo", options=["call", "put"], required=True),
                    "strike": st.column_config.NumberColumn("Strike (K)", format="$%.2f", min_value=0.1, required=True),
                    "premium": st.column_config.NumberColumn("Prima (Premium)", format="$%.2f", min_value=0.0, required=True),
                    "qty": st.column_config.NumberColumn("Contratos (Qty)", min_value=1, default=1, required=True),
                },
                width='stretch'
            )
            
            if edited_legs:
                # Generar rango de precios para el eje X alrededor del Spot (desde -40% hasta +40%)
                s_t = np.linspace(spot_price * 0.6, spot_price * 1.4, 200)
                
                try:
                    # Calcular el Payoff global
                    total_pnl = calculate_strategy_payoff(s_t, edited_legs)
                    
                    # Calcular métricas resumen globales de la estrategia
                    max_profit = np.max(total_pnl)
                    max_loss = np.min(total_pnl)
                    
                    # Renderizar métricas informativas en tarjetas
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Máxima Ganancia Teórica", f"${max_profit:.2f}" if max_profit < 999999 else "Ilimitada")
                    c2.metric("Máximo Riesgo Teórico", f"${abs(max_loss):.2f}" if max_loss > -999999 else "Ilimitado")
                    
                    # Determinar el costo/crédito neto inicial del spread
                    net_premium = sum(
                        (leg['premium'] if leg['action'] == 'buy' else -leg['premium']) * leg.get('qty', 1)
                        for leg in edited_legs if 'premium' in leg and 'action' in leg
                    )
                    label_premium = "Débito Neto (Costo)" if net_premium >= 0 else "Crédito Neto (Ingreso)"
                    c3.metric(label_premium, f"${abs(net_premium * 100):.2f}") # Multiplicado por 100 multiplicador estándar de opciones
                    
                    # Renderizar gráfico Plotly 2D
                    fig_payoff = plot_payoff_profile(s_t, total_pnl, spot_price)
                    st.plotly_chart(fig_payoff, width='stretch')
                    
                except Exception as math_error:
                    st.info(":material/brightness_alert: Asegúrate de rellenar completamente todos los campos numéricos de la tabla para proyectar el riesgo.")

    # --- MÓDULO 10: MACHINE LEARNING & ARBITRAJE ---
        elif modulo == "Machine Learning Predictivo":
            st.subheader(f"AI Intelligence & Arbitrage Center: {ticker_input}")
            st.markdown("Plataforma analítica con modelos predictivos y escáner estadístico de ineficiencias de mercado.")
            
            # Creación de pestañas estéticas
            tab1, tab2, tab3 = st.tabs([
                ":material/assistant_direction: Dirección de IV (Clasificación)", 
                ":material/brightness_empty: Valor Numérico de IV (Regresión)", 
                ":material/nest_detect: Detector de Anomalías (Arbitraje)"
            ])
            
            # Descarga de datos única compartida para los modelos predictivos de la acción
            with st.spinner("Sincronizando modelos cuantitativos en tiempo real..."):
                df_hist = yf.download(ticker_input, period="1y", progress=False)
                if not df_hist.empty and isinstance(df_hist.columns, pd.MultiIndex):
                    df_hist.columns = df_hist.columns.get_level_values(0)

            if df_hist.empty:
                st.error(":material/error: No se pudieron inicializar los modelos por falta de datos históricos.")
            else:
                # --- TAB 1: MODELO ORIGINAL DE CLASIFICACIÓN ---
                with tab1:
                    st.subheader("Predicción de Dirección de la Volatilidad Implícita")
                    st.caption(":material/save_as: Clasificador Random Forest enfocado en determinar la dirección binaria (Subida/Caída) a 5 días de horizonte.")
                    
                    predictor = IVDirectionPredictor(model_type='random_forest', n_estimators=100)
                    df_features = predictor.engineer_features(df_hist, target_horizon=5)
                    metrics = predictor.train(df_features)
                    pred_class, prob_up = predictor.predict_latest(df_features)
                    
                    col_m1, col_m2 = st.columns(2)
                    signal_text = "ALCISTA (La IV subirá)" if pred_class == 1 else "BAJISTA / LATERAL (La IV caerá o se mantendrá)"
                    signal_color = "normal" if pred_class == 1 else "inverse"
                    
                    col_m1.metric("Dirección Proyectada (Próximos 5 días)", signal_text, delta=f"Probabilidad: {prob_up*100:.1f}%", delta_color=signal_color)
                    col_m2.metric("Precisión del Modelo (Accuracy)", f"{metrics['Accuracy']*100:.1f}%")
                    
                    st.divider()
                    st.subheader("Métricas de Evaluación de Clasificación")
                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Precision", f"{metrics['Precision']:.2f}")
                    met2.metric("Recall", f"{metrics['Recall']:.2f}")
                    met3.metric("F1-Score", f"{metrics['F1_Score']:.2f}")
                    met4.metric("ROC AUC", f"{metrics['ROC_AUC']:.2f}")
                    
                    importances = predictor.get_feature_importances()
                    df_imp = pd.DataFrame(list(importances.items()), columns=['Variable', 'Importancia']).sort_values(by='Importancia', ascending=True)
                    fig_imp = go.Figure(go.Bar(x=df_imp['Importancia'], y=df_imp['Variable'], orientation='h', marker_color='#deff9a'))
                    fig_imp.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_imp, width='stretch')

                # --- TAB 2: NUEVO MODELO DE REGRESIÓN DE VOLATILIDAD ---
                with tab2:
                    st.subheader("Pronóstico Numérico del Régimen de Volatilidad Futuro")
                    st.caption(":material/save_as: Regressor avanzado encargado de mapear y predecir el número exacto del régimen de volatilidad intradía.")
                    
                    regressor = IVValueRegressor(n_estimators=100)
                    df_feat_reg = regressor.engineer_features(df_hist, target_horizon=5)
                    metrics_reg = regressor.train(df_feat_reg)
                    
                    predicted_val = regressor.predict_latest(df_feat_reg)
                    current_val = df_feat_reg['Parkinson_Proxy'].iloc[-1]
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Volatilidad Proyectada (5 días)", f"{predicted_val*100:.2f}%", delta=f"{((predicted_val - current_val)*100):+.2f}% vs Actual")
                    r2.metric("Error Medio Absoluto (MAE)", f"{metrics_reg['MAE']*100:.2f}%", help="Desviación promedio esperada del modelo numérico.")
                    r3.metric("Bondad de Ajuste (R² Score)", f"{metrics_reg['R2']:.2f}", help="Puntuación de correlación. Cerca de 1.0 es un ajuste perfecto.")
                    
                    st.divider()
                    st.subheader("Importancia de Variables (Regresor)")
                    imp_reg = regressor.get_feature_importances()
                    df_imp_reg = pd.DataFrame(list(imp_reg.items()), columns=['Variable', 'Importancia']).sort_values(by='Importancia', ascending=True)
                    fig_imp_reg = go.Figure(go.Bar(x=df_imp_reg['Importancia'], y=df_imp_reg['Variable'], orientation='h', marker_color='#38bdf8'))
                    fig_imp_reg.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_imp_reg, width='stretch')

                # --- TAB 3: NUEVO DETECTOR DE ANOMALÍAS DE PRECIO (ARBITRAJE) ---
                with tab3:
                    st.subheader("Escáner de Arbitraje Estadístico sobre Opciones de Mercado")
                    st.markdown("Evalúa desviaciones extremas entre el valor cotizado en vivo y el modelo matemático de Black-Scholes.")
                    
                    if df_options.empty:
                        st.warning(":material/warning: No hay contratos de opciones disponibles para escanear.")
                    else:
                        # Filtro de fecha específico para aislar el arbitraje relativo por maduración
                        fechas_anom = sorted(df_options['expiration_date'].dt.strftime('%Y-%m-%d').unique())
                        selected_date_anom = st.selectbox("Seleccionar Expiración del Contrato", fechas_anom, key="date_anom")
                        
                        df_slice = df_options[df_options['expiration_date'].dt.strftime('%Y-%m-%d') == selected_date_anom].copy()
                        
                        with st.spinner("Ejecutando escaneo matricial de primas..."):
                            greeks_data = []
                            for _, row in df_slice.iterrows():
                                bsm = BlackScholesModel(S=spot_price, K=row['strike'], T=row['dte'], r=risk_free, sigma=row['impliedVolatility'])
                                greeks_data.append({
                                    'Strike': row['strike'],
                                    'Tipo': row['option_type'].upper(),
                                    'lastPrice': row['lastPrice'],
                                    'Theoretical Price': bsm.price(option_type=row['option_type'])
                                })
                            df_calc = pd.DataFrame(greeks_data)
                            
                            z_thresh = st.slider("Umbral de Sensibilidad Estadística (Z-Score)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
                            df_anomalies = detect_price_anomalies(df_calc, z_threshold=z_thresh)
                            
                            anomalies_found = df_anomalies[df_anomalies['Alerta_Sujeta'] != 'Normal']
                            
                            st.metric("Oportunidades de Arbitraje Encontradas", len(anomalies_found), delta=f"De {len(df_slice)} contratos evaluados")
                            
                            if not anomalies_found.empty:
                                st.success(f":material/bookmark_check: Se detectaron {len(anomalies_found)} anomalías de precio significativas.")
                                
                                def style_anomalies(val):
                                    if 'Subvaluada' in str(val): return 'background-color: rgba(34, 197, 94, 0.15); color: #22c55e; font-weight: bold;'
                                    if 'Sobrevaluada' in str(val): return 'background-color: rgba(239, 68, 68, 0.15); color: #ef4444; font-weight: bold;'
                                    return ''
                                
                                styled_anom = anomalies_found.sort_values(by='Z_Score').style.map(style_anomalies, subset=['Alerta_Sujeta']).format({
                                    'lastPrice': '${:.2f}',
                                    'Theoretical Price': '${:.2f}',
                                    'Price_Dev_Pct': '{:.1%}',
                                    'Z_Score': '{:.2f}'
                                })
                                st.dataframe(styled_anom, width='stretch')
                            else:
                                st.info(":material/warning: No se hallaron ineficiencias críticas con este umbral de sensibilidad. La cadena se encuentra balanceada.")

        # --- MÓDULO 5: QUANTUM FINANCE ---
        elif modulo == "Quantum Finance":
            st.subheader(f"Laboratorio Cuántico: {ticker_input}")
            st.markdown("> **Nota de Investigación:** Este módulo utiliza el simulador `Statevector` de Qiskit para mapear la distribución lognormal del activo a estados cuánticos.")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Parámetros del Circuito")
                qubits = st.slider("Qubits (Resolución)", min_value=2, max_value=8, value=5, help="Más qubits = mayor resolución (2^n estados computacionales).")
                strike_q = st.number_input("Strike Price (K)", value=round(spot_price, 2))
                dte_q = st.number_input("Días al Vencimiento (DTE)", value=30.0) / 365.25
                
                default_vol = df_options['impliedVolatility'].mean() if not df_options.empty else 0.20
                vol_q = st.number_input("Volatilidad Implícita", value=float(default_vol))
                
                run_qmc = st.button("Ejecutar Quantum Monte Carlo", type="primary", width='stretch')          

            with col2:
                if run_qmc:
                    with st.spinner(f"Inicializando simulador cuántico con {qubits} qubits..."):
                        try:
                            # 1. Inicialización y Ejecución del Modelo Cuántico
                            pricer = QuantumMonteCarloPricer(
                                spot=spot_price, 
                                strike=strike_q, 
                                rate=risk_free, 
                                vol=vol_q, 
                                dte=dte_q, 
                                num_qubits=qubits
                            )
                            
                            resultados = pricer.price_european_call()
                            S_T, probs = pricer._get_lognormal_distribution()
                            
                            # 2. CÁLCULOS TRADICIONALES DE CONTROL (NUEVA INCORPORACIÓN)
                            classic_mc_p = classic_monte_carlo_price(
                                spot=spot_price, strike=strike_q, rate=risk_free, 
                                vol=vol_q, dte=dte_q, paths=100000, option_type='call'
                            )
                            
                            binomial_euro_p = binomial_tree_price(
                                spot=spot_price, strike=strike_q, rate=risk_free, 
                                vol=vol_q, dte=dte_q, steps=150, option_type='call', exercise_type='european'
                            )
                            
                            binomial_amer_p = binomial_tree_price(
                                spot=spot_price, strike=strike_q, rate=risk_free, 
                                vol=vol_q, dte=dte_q, steps=150, option_type='call', exercise_type='american'
                            )
                            
                            # 3. RENDERIZADO INTERFAZ VISUAL
                            st.success("¡Simulación de Vectores de Estado Completada con Qiskit!")
                            
                            # Fila A: Métricas Técnicas del Simulador Cuántico
                            st.markdown(":material/settings: Especificaciones de la Simulación Cuántica")
                            met1, met2, met3 = st.columns(3)
                            met1.metric("Precio Call Cuántico", f"${resultados['quantum_price']:.4f}")
                            met2.metric("Qubits Activos", resultados['qubits_used'])
                            met3.metric("Estados Superpuestos (2^n)", resultados['quantum_states'])
                            
                            st.divider()
                            
                            # Fila B: Módulo de Arbitraje y Benchmark Matemático
                            st.markdown(":material/finance: Benchmark de Modelos Matemáticos")
                            st.markdown("Contraste científico del valor esperado cuántico frente a los estándares de la industria.")
                            
                            b1, b2, b3, b4 = st.columns(4)
                            b1.metric("Simulador Cuántico", f"${resultados['quantum_price']:.4f}", help="Calculado en superposición cuántica.")
                            b2.metric("Monte Carlo Clásico (100k)", f"${classic_mc_p:.4f}", help="Promedio de 100,000 caminos aleatorios clásicos.")
                            b3.metric("Árbol Binomial (Europea)", f"${binomial_euro_p:.4f}", help="Modelo CRR sin ejercicio anticipado.")
                            b4.metric("Árbol Binomial (Americana)", f"${binomial_amer_p:.4f}", help="Modelo CRR con derecho a ejercicio anticipado.")
                            
                            # Análisis cuantitativo de la prima de ejercicio anticipado
                            early_exercise_premium = binomial_amer_p - binomial_euro_p
                            if early_exercise_premium > 0.001:
                                st.info(f":material/chat_info: **Análisis Quanti:** La versión **Americana** de esta opción cotiza con una prima de **${early_exercise_premium:.4f}** por encima de la Europea debido a la probabilidad de ejercicio anticipado óptimo.")
                            
                            st.divider()

                            # Fila C: Gráfico de Amplitudes Cuánticas
                            fig_q = go.Figure()
                            fig_q.add_trace(go.Bar(
                                x=np.round(S_T, 2), 
                                y=probs, 
                                marker_color='#a3e635',
                                name='Amplitud^2 (Probabilidad)'
                            ))
                            fig_q.add_vline(x=strike_q, line_dash="dash", line_color="#ef4444", annotation_text="Strike")
                            fig_q.add_vline(x=spot_price, line_dash="solid", line_color="#f8fafc", annotation_text="Spot Actual")
                            
                            fig_q.update_layout(
                                title="Distribución Mapeada al Registro Cuántico",
                                xaxis_title="Precio del Activo en el Vencimiento",
                                yaxis_title="Probabilidad",
                                template="plotly_dark",
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_q, width='stretch')
                            
                        except Exception as e:
                            st.error(f"Error en la ejecución cuántica: {e}")
                else:
                    st.info("Configura los parámetros y presiona 'Ejecutar Quantum Monte Carlo' para iniciar el simulador.")

        else:
            st.subheader(f"Módulo: {modulo}")
            st.info("Este módulo está diseñado en la arquitectura pero requiere la integración de librerías avanzadas (Qiskit/Scikit-Learn) según la estructura desarrollada.")

    except Exception as e:
        st.error(f"Error al inicializar la plataforma: {e}")


