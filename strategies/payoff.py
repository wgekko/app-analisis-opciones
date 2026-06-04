import numpy as np
from typing import List, Dict

def calculate_leg_payoff(s_t: np.ndarray, leg_type: str, action: str, strike: float, premium: float, qty: int = 1) -> np.ndarray:
    """
    Calcula el P&L neto de una sola 'pata' (leg) de la opción en el vencimiento (S_T).
    """
    # 1. Calcular Payoff Bruto al vencimiento
    if leg_type.lower() == 'call':
        payoff = np.maximum(s_t - strike, 0)
    elif leg_type.lower() == 'put':
        payoff = np.maximum(strike - s_t, 0)
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")
    
    # 2. Ajustar según la acción de mercado (Compra / Venta Corta) e incluir prima
    if action.lower() == 'buy':
        pnl = (payoff - premium) * qty
    elif action.lower() == 'sell':
        pnl = (premium - payoff) * qty
    else:
        raise ValueError("La acción debe ser 'buy' o 'sell'")
        
    return pnl

def calculate_strategy_payoff(s_t: np.ndarray, legs: List[Dict]) -> np.ndarray:
    """
    Consolida algebraicamente el P&L acumulado de todas las patas del spread.
    """
    total_pnl = np.zeros_like(s_t)
    for leg in legs:
        total_pnl += calculate_leg_payoff(
            s_t=s_t,
            leg_type=leg['type'],
            action=leg['action'],
            strike=float(leg['strike']),
            premium=float(leg['premium']),
            qty=int(leg.get('qty', 1))
        )
    return total_pnl