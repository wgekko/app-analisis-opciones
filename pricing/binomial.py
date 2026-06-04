import numpy as np

def binomial_tree_price(spot: float, strike: float, rate: float, vol: float, dte: float, steps: int = 100, option_type: str = 'call', exercise_type: str = 'american') -> float:
    """
    Valúa opciones europeas o americanas mediante el Árbol Binomial de Cox-Ross-Rubinstein (CRR).
    Optimizado de forma matricial (vectorizado) para alta velocidad en Streamlit.
    """
    if dte <= 0 or vol <= 0 or steps <= 0:
        return 0.0

    dt = dte / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(rate * dt) - d) / (u - d)
    
    # Validar condición de no arbitraje (probabilidad martingale entre 0 y 1)
    if not (0 <= p <= 1):
        # Si la tasa o volatilidad son extremas, forzar aproximación segura
        p = max(0.0, min(1.0, p))

    # 1. Inicializar los precios del activo subyacente en el vencimiento (Paso N)
    s = spot * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
    
    # 2. Inicializar los valores de la opción en el vencimiento
    if option_type.lower() == 'call':
        v = np.maximum(s - strike, 0)
    elif option_type.lower() == 'put':
        v = np.maximum(strike - s, 0)
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")
        
    # 3. Inducción hacia atrás recorriendo los niveles del árbol
    disc = np.exp(-rate * dt)
    for i in range(steps - 1, -1, -1):
        # Precios del subyacente en el paso actual 'i'
        s = spot * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
        
        # Valor de continuación (teórico europeo descontado)
        v_continuation = disc * (p * v[:-1] + (1.0 - p) * v[1:])
        
        if exercise_type.lower() == 'american':
            # Si es americana, elegimos el máximo entre esperar (continuar) o ejercer hoy
            if option_type.lower() == 'call':
                v_intrinsic = np.maximum(s - strike, 0)
            else:
                v_intrinsic = np.maximum(strike - s, 0)
            v = np.maximum(v_continuation, v_intrinsic)
        else:
            v = v_continuation
            
    return float(v[0])