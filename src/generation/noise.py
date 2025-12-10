import numpy as np
from scipy.signal import lfilter
from typing import Tuple

class ColoredNoise:
    """
    Generates colored noise (AR(1) process) and baseline drift.
    """
    def __init__(self, level: float, color_factor: float, baseline_drift: float):
        """
        Args:
            level: Standard deviation of the white noise component.
            color_factor: AR(1) coefficient (alpha). 0 = white noise, close to 1 = red noise.
            baseline_drift: Magnitude of the low-frequency baseline drift.
        """
        self.level = level
        self.color_factor = color_factor
        self.baseline_drift = baseline_drift

    def generate(self, shape: Tuple[int, int], seed: int = None) -> np.ndarray:
        """
        Generates noise matrix of given shape.
        
        Args:
            shape: (n_traces, length)
            seed: Random seed.
            
        Returns:
            np.ndarray: Noise matrix.
        """
        rng = np.random.default_rng(seed)
        n_traces, length = shape

        # 1. Generate White Noise
        white = rng.normal(0, self.level, size=shape)

        # 2. Apply AR(1) filter for Colored Noise
        # y[n] = alpha * y[n-1] + x[n] -> y[n] - alpha * y[n-1] = x[n]
        # Transfer function: H(z) = 1 / (1 - alpha * z^-1)
        # b = [1], a = [1, -alpha]
        if self.color_factor != 0:
            b = [1.0]
            a = [1.0, -self.color_factor]
            # lfilter applies along the last axis by default (axis=-1), which is 'length'
            colored = lfilter(b, a, white, axis=-1)
        else:
            colored = white

        # 3. Add Baseline Drift
        # Simulate drift as a very low frequency sine wave or random walk
        # Here we use a simplified random walk approach for each trace
        if self.baseline_drift > 0:
             # Generate random walk: cumulative sum of small random steps
            drift_steps = rng.normal(0, self.baseline_drift / np.sqrt(length), size=shape)
            drift = np.cumsum(drift_steps, axis=1)
            # Center the drift to avoid massive offsets
            drift -= np.mean(drift, axis=1, keepdims=True)
            colored += drift

        return colored
