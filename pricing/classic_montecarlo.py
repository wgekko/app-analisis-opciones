import numpy as np

def classic_monte_carlo_price(spot: float, strike: float, rate: float, vol: float, dte: float, paths: int = 100000, option_type: str = 'call') -> float:
    """
    Valúa una opción europea (Call o Put) mediante la simulación clásica de Monte Carlo
    usando Movimiento Browniano Geométrico estándar.
    """
    if dte <= 0 or vol <= 0:
        return 0.0
        
    # Fijar semilla para reproducibilidad científica en la terminal
    np.random.seed(42)
    
    # Generar variables normales estándar Z ~ N(0, 1)
    z = np.random.standard_normal(paths)
    
    # Simular el precio final S_T mediante la ecuación diferencial estocástica integrada
    s_t = spot * np.exp((rate - 0.5 * vol**2) * dte + vol * np.sqrt(dte) * z)
    
    # Evaluar función de Payoff según el tipo de contrato
    if option_type.lower() == 'call':
        payoffs = np.maximum(s_t - strike, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(strike - s_t, 0)
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")
        
    # Descontar el valor esperado a valor presente (t=0)
    price = np.exp(-rate * dte) * np.mean(payoffs)
    return float(price)