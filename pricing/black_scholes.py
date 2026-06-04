import numpy as np
from scipy.stats import norm
from typing import Dict

class BlackScholesModel:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        S: Precio Spot
        K: Precio Strike
        T: Tiempo al vencimiento (en años)
        r: Tasa libre de riesgo
        sigma: Volatilidad implícita
        q: Dividend yield
        """
        self.S = S
        self.K = K
        self.T = max(T, 1e-5) # Evitar división por cero
        self.r = r
        self.sigma = max(sigma, 1e-5)
        self.q = q
        
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price(self, option_type: str = 'call') -> float:
        if option_type == 'call':
            return (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - 
                    self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        elif option_type == 'put':
            return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                    self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
        raise ValueError("option_type debe ser 'call' o 'put'")

    def greeks(self, option_type: str = 'call') -> Dict[str, float]:
        pdf_d1 = norm.pdf(self.d1)
        
        delta = norm.cdf(self.d1) if option_type == 'call' else norm.cdf(self.d1) - 1
        gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * pdf_d1 * np.sqrt(self.T) / 100 # Escala estándar de mercado (1%)
        
        if option_type == 'call':
            theta = (- (self.S * self.sigma * pdf_d1) / (2 * np.sqrt(self.T)) 
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 365
            rho = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 100
        else:
            theta = (- (self.S * self.sigma * pdf_d1) / (2 * np.sqrt(self.T)) 
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 365
            rho = (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 100
            
        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho,
            "Vanna": self.vanna(),  # <-- ENLACE REQUERIDO PARA TU APP.PY
            "Volga": self.volga()   # <-- ENLACE REQUERIDO PARA TU APP.PY
        }

    def vanna(self) -> float:
        """
        Mide la sensibilidad de la Delta respecto a cambios en la Volatilidad Implícita (IV).
        Escalada para reflejar el cambio en Delta por cada 1% de movimiento en IV.
        """
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
            
        pdf_d1 = norm.pdf(self.d1)
        # Reutiliza de forma óptima las propiedades calculadas en el constructor
        vanna_val = (- np.exp(-self.q * self.T) * pdf_d1 * self.d2 / self.sigma) / 100
        return float(vanna_val)

    def volga(self) -> float:
        """
        Mide la aceleración de la Vega respecto a cambios en la Volatilidad Implícita (IV).
        Escalada para reflejar el cambio en Vega por cada 1% de movimiento en IV.
        """
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
            
        pdf_d1 = norm.pdf(self.d1)
        vega_raw = self.S * np.exp(-self.q * self.T) * pdf_d1 * np.sqrt(self.T)
        
        # Escala de mercado institucional de segundo orden
        volga_val = (vega_raw * self.d1 * self.d2 / self.sigma) / 10000
        return float(volga_val)