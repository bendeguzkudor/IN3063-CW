import random
import numpy as np

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across all random number generators.
    Args:
        seed (int): Seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)