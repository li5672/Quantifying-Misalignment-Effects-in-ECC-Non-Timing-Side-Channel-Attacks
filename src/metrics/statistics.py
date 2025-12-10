import numpy as np
from typing import Tuple

def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes the Wilson Score Interval for a binomial proportion.
    
    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence: Confidence level (default 0.95).
        
    Returns:
        (center, lower, upper): The estimated proportion and the confidence interval bounds.
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
        
    z = 1.96 # Approx for 95% confidence
    if confidence != 0.95:
        # For other confidence levels, we would need scipy.stats.norm.ppf
        # But to keep dependencies minimal in this utility, we default to 1.96
        # If needed, we can import scipy.stats
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center_adjusted = (p_hat + z**2 / (2 * trials)) / denominator
    
    term = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials)
    lower = (center_adjusted - term / denominator)
    upper = (center_adjusted + term / denominator)
    
    # Clip to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    
    return p_hat, lower, upper
