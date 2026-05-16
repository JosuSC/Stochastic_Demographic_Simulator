import random
from typing import List, Tuple

def get_uniform(min_val: float, max_val: float) -> float:
    """Devuelve un valor de distribución uniforme."""
    return random.uniform(min_val, max_val)

def get_exponential(mean: float) -> float:
    """Devuelve un valor de distribución exponencial con la media dada.
    
    Usa random.expovariate(lambd) donde lambd = 1/mean,
    lo cual produce una distribución con la media especificada.
    """
    if mean <= 0:
        return 0.0
    return random.expovariate(1.0 / mean)

def evaluate_probability(prob: float) -> bool:
    """Evalúa un suceso de Bernoulli con probabilidad prob."""
    return random.random() < prob
