# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Reproducibility utilities for setting random seeds across all libraries.

Author: Hachem
Created: Fri Jan 2 00:20:45 2026
"""

import random
import numpy as np
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value
        
    Notes
    -----
    Sets seeds for:
    - Python's random module
    - NumPy
    - PYTHONHASHSEED (for hash randomization)
    - XGBoost (via n_jobs=1 or seed parameter in model)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Note: XGBoost seed must be set in model parameters
    # as random_state=seed when initializing XGBRegressor
    
    print(f"[SEED] Random seed set to {seed} for reproducibility")


def get_random_state(config):
    """
    Extract random seed from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    int
        Random seed value
    """
    return config.get("experiment", {}).get("random_seed", 42)