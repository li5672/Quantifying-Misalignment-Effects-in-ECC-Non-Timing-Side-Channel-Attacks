import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d

class MetricCalculator:
    """
    Calculates N80 and Jitter Degradation Factor r(j).
    """
    
    @staticmethod
    def compute_n80(n_list: List[int], success_rates: List[float]) -> float:
        """
        Interpolates to find the number of traces N required for 80% success rate.
        
        Args:
            n_list: List of N values (trace counts).
            success_rates: Corresponding success rates (0.0 to 1.0).
            
        Returns:
            n80: Interpolated N value. Returns inf if 80% is not reached.
        """
        n_arr = np.array(n_list)
        s_arr = np.array(success_rates)
        
        # If max success < 0.8, we cannot estimate N80
        if np.max(s_arr) < 0.8:
            return float('inf')
            
        # If min success > 0.8, N80 is likely smaller than min(n_list)
        if np.min(s_arr) > 0.8:
            # Extrapolate or return min?
            # Let's return min for safety, or linear extrapolation to 0?
            # Let's just return the first N where S > 0.8
            return float(n_arr[0])
            
        # Interpolation
        # We want to find n such that S(n) = 0.8
        # S(n) is roughly monotonic increasing.
        
        f = interp1d(s_arr, n_arr, kind='linear', bounds_error=False, fill_value="extrapolate")
        n80 = f(0.8)
        
        return float(n80)

    @staticmethod
    def compute_jitter_degradation(n80_jitter: float, n80_baseline: float) -> float:
        """
        Computes r(j) = N80(j) / N80(0).
        """
        if n80_baseline == 0:
            return float('inf')
        if n80_jitter == float('inf'):
            return float('inf')
            
        return n80_jitter / n80_baseline
