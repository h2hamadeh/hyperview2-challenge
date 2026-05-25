
import logging
import random
import numpy as np
import os

logger = logging.getLogger("hyperview2")


def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"[SEED] Random seed set to {seed} for reproducibility")


def get_random_state(config):
    
    return config.get("experiment", {}).get("random_seed", 42)