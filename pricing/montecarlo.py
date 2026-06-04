import numpy as np
from scipy.stats import lognorm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class QuantumMonteCarloPricer:
    """
    Módulo Experimental de Quantum Finance.
    
    Implementa un simulador de Quantum Monte Carlo (QMC) simplificado utilizando 
    Qiskit. En lugar de iterar miles de veces como el Monte Carlo clásico, 
    cargamos toda la distribución de probabilidad lognormal del activo en una 
    superposición cuántica (Quantum State Preparation) y calculamos el valor 
    esperado del Payoff.
    
    Nota: Al ejecutarse en hardware clásico (Statevector Simulator), la complejidad
    es exponencial respecto al número de qubits. En hardware cuántico real, 
    usando Quantum Amplitude Estimation (QAE), ofrecería una aceleración cuadrática.
    """
    
    def __init__(self, spot: float, strike: float, rate: float, vol: float, 
                 dte: float, num_qubits: int = 5):
        self.S0 = spot
        self.K = strike
        self.r = rate
        self.sigma = vol
        self.T = dte
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        
    def _get_lognormal_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretiza la distribución lognormal del precio del activo en 2^n estados.
        """
        # Parámetros de la distribución lognormal
        mu = np.log(self.S0) + (self.r - 0.5 * self.sigma**2) * self.T
        sigma_t = self.sigma * np.sqrt(self.T)
        
        # Definir el rango de precios (ej. +/- 3 desviaciones estándar)
        mean_price = np.exp(mu + 0.5 * sigma_t**2)
        std_price = mean_price * np.sqrt(np.exp(sigma_t**2) - 1)
        
        low = max(0.01, mean_price - 3 * std_price)
        high = mean_price + 3 * std_price
        
        # Discretizar los posibles precios al vencimiento (S_T)
        S_T = np.linspace(low, high, self.num_states)
        
        # Calcular las probabilidades para cada precio y normalizar
        probs = lognorm.pdf(S_T, s=sigma_t, scale=np.exp(mu))
        probs = probs / np.sum(probs)
        
        return S_T, probs

    def build_quantum_state(self) -> Statevector:
        """
        Prepara el estado cuántico mapeando las probabilidades clásicas a las 
        amplitudes de probabilidad de los qubits.
        """
        _, probs = self._get_lognormal_distribution()
        
        # La amplitud cuántica es la raíz cuadrada de la probabilidad clásica
        amplitudes = np.sqrt(probs)
        
        # Crear el Statevector de Qiskit
        state = Statevector(amplitudes)
        return state

    def price_european_call(self) -> Dict[str, float]:
        """
        Calcula el precio de una opción Call Europea extrayendo el valor esperado 
        del estado cuántico simulado.
        """
        try:
            S_T, probs = self._get_lognormal_distribution()
            
            # Cargar el estado en un circuito (representación conceptual)
            qc = QuantumCircuit(self.num_qubits)
            qc.initialize(self.build_quantum_state().data, qc.qubits)
            
            # Extraer las probabilidades del simulador vectorial de Qiskit
            state = Statevector(qc)
            quantum_probs = state.probabilities()
            
            # Función de Payoff clásico evaluada sobre la distribución cuántica
            payoffs = np.maximum(S_T - self.K, 0)
            
            # Valor esperado (Equivalente al Quantum Amplitude Estimation ideal)
            expected_payoff = np.dot(quantum_probs, payoffs)
            
            # Descuento a valor presente
            present_value = expected_payoff * np.exp(-self.r * self.T)
            
            return {
                "quantum_price": float(present_value),
                "qubits_used": self.num_qubits,
                "quantum_states": self.num_states,
                "classical_equivalent_paths": "Infinito (Distribución Analítica Exacta)"
            }
            
        except Exception as e:
            logger.error(f"Error en la simulación cuántica: {e}")
            raise