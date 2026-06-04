
## Quant Options Analytics Terminal 

Una plataforma analítica e interactiva de grado institucional para el modelado, pricing y gestión de riesgos en el mercado de opciones financieras. Desarrollada íntegramente en Python, esta terminal unifica modelos analíticos matemáticos clásicos, algoritmos de Machine Learning para volatilidad implícita (IV) y un módulo experimental de Finanzas Cuánticas (*Quantum Finance*).

## Características Principales

**Option Chain Analyzer:** Conexión en tiempo real a flujos de datos de mercado (Yahoo Finance) con visualización dinámica de la superficie de volatilidad en 3D.
**Advanced Greeks Engine:** Cálculo preciso de sensibilidades de primer orden (Delta, Gamma, Vega, Theta, Rho) y métricas avanzadas de segundo orden de nivel creador de mercado (**Vanna y Volga**) escaladas para consistencia institucional.
**Multi-Model Pricing Engine:** Evaluación de primas mediante Black-Scholes, Árboles Binomiales, Simulación Monte Carlo Clásica y Arbitraje Estadístico.
**Machine Learning Module:** Modelos predictivos y de regresión para la dirección y valor proyectado de la IV, acompañados de un detector de anomalías de precio (Escáner de Arbitraje).
**Quantum Monte Carlo (QMC):** Módulo experimental que utiliza **Qiskit** para la preparación de estados cuánticos a partir de distribuciones lognormales, simulando el cálculo del valor esperado del *payoff* de opciones de manera nativa en superposición.

## Estructura del Proyecto

├── app.py                      # Interfaz principal de la terminal (Streamlit)
├── requirements.txt            # Dependencias del sistema
├── data/
│   └── yfinance_provider.py    # Proveedor y caché de datos de mercado
├── pricing/
│   ├── black_scholes.py        # Modelo analítico y motor de Griegas (1° y 2° orden)
│   ├── classic_montecarlo.py   # Simulación estocástica clásica
│   ├── binomial.py             # Modelo de redes binomiales Cox-Ross-Rubinstein
│   └── montecarlo.py           # Simulador cuántico (Quantum Monte Carlo Pricer)
├── ml_models/
│   ├── iv_predictor.py         # Clasificador direccional de Volatilidad
│   ├── iv_regressor.py         # Regresor para estimación cuantitativa de IV
│   └── anomaly_detector.py     # Filtro estadístico y Z-Score para arbitraje
├── components/
│   ├── charts.py               # Renderizado de superficies 3D y curvas de Payoff (Plotly)
│   └── tables.py               # Formateo avanzado y mapas de calor interactivos
└── strategies/
    ├── builder.py              # Estructuras lógicas de estrategias multi-leg
    └── payoff.py               # Calculador analítico de perfiles de riesgo

git clone [https://github.com/tu-usuario/quant-options-analytics.git](https://github.com/tu-usuario/quant-options-analytics.git)
cd quant-options-analytics
