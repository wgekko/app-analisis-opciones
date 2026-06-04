from typing import List, Dict

def get_strategy_presets(spot: float) -> Dict[str, List[Dict]]:
    """
    Retorna estructuras predefinidas de estrategias multi-leg 
    ajustadas dinámicamente según el precio spot actual.
    """
    s = round(spot) # strike base central
    
    return {
        "Personalizada (Crear desde cero)": [],
        
        "Bull Call Spread (Alcista)": [
            {"action": "buy", "type": "call", "strike": s - 2, "premium": 3.50, "qty": 1},
            {"action": "sell", "type": "call", "strike": s + 3, "premium": 1.20, "qty": 1}
        ],
        
        "Bear Put Spread (Bajista)": [
            {"action": "buy", "type": "put", "strike": s + 2, "premium": 3.20, "qty": 1},
            {"action": "sell", "type": "put", "strike": s - 3, "premium": 1.10, "qty": 1}
        ],
        
        "Straddle Largo (Alta Volatilidad)": [
            {"action": "buy", "type": "call", "strike": s, "premium": 2.80, "qty": 1},
            {"action": "buy", "type": "put", "strike": s, "premium": 2.50, "qty": 1}
        ],
        
        "Iron Condor (Rango Lateral)": [
            {"action": "buy", "type": "put", "strike": s - 10, "premium": 0.40, "qty": 1},
            {"action": "sell", "type": "put", "strike": s - 5, "premium": 1.50, "qty": 1},
            {"action": "sell", "type": "call", "strike": s + 5, "premium": 1.45, "qty": 1},
            {"action": "buy", "type": "call", "strike": s + 10, "premium": 0.35, "qty": 1}
        ]
    }