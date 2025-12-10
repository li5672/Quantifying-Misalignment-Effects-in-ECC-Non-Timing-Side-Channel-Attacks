import numpy as np
from typing import Tuple, List
from ecc_sca_project.src.metrics.pearson import vectorized_correlation

class CPAAttack:
    """
    Correlation Power Analysis (CPA) Attack.
    Recovers bits by correlating Hamming Weight/Distance predictions with power traces.
    """
    def __init__(self, pulse_len: int, gap: int):
        self.pulse_len = pulse_len
        self.gap = gap

    def _otsu_threshold(self, values: np.ndarray) -> float:
        """
        Computes the optimal threshold using Otsu's Method.
        """
        hist, bin_edges = np.histogram(values, bins=256, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist = hist / np.sum(hist)
        cum_sum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * bin_centers)
        global_mean = cum_mean[-1]
        
        valid_mask = (cum_sum > 0) & (cum_sum < 1)
        numerator = (global_mean * cum_sum[valid_mask] - cum_mean[valid_mask]) ** 2
        denominator = cum_sum[valid_mask] * (1.0 - cum_sum[valid_mask])
        variance = numerator / denominator
        
        max_var = np.max(variance)
        optimal_indices = np.where(variance >= max_var * 0.9999)[0]
        valid_indices_in_original = np.where(valid_mask)[0]
        best_original_indices = valid_indices_in_original[optimal_indices]
        best_threshold = np.mean(bin_centers[best_original_indices])
        
        return best_threshold

    def recover_key(self, traces: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recovers the key (bits) from a set of traces.
        Assumes we know the start positions (aligned attack) or use nominal positions.
        
        Args:
            traces: (n_traces, n_samples)
            starts: (n_traces, n_rounds)
            
        Returns:
            bits: (n_rounds,) Recovered bits.
        """
        n_traces, n_samples = traces.shape
        n_traces_starts, n_rounds = starts.shape
        
        if n_traces != n_traces_starts:
            raise ValueError("Mismatch in number of traces between traces and starts")
            
        # Ideal Add Pulse
        x = np.arange(self.pulse_len)
        center = self.pulse_len // 2
        width = self.pulse_len / 6
        template = np.exp(-0.5 * ((x - center) / width) ** 2)
        
        all_correlations = []
        
        # Iterate through each round (bit)
        for r in range(n_rounds):
            # Extract windows for all traces
            s_add = starts[:, r] + self.pulse_len + self.gap
            
            # We need to extract (n_traces, pulse_len) matrix
            # Advanced indexing
            # indices: (n_traces, pulse_len)
            indices = s_add[:, np.newaxis] + np.arange(self.pulse_len)
            
            # Handle bounds
            # Clip indices
            indices = np.clip(indices, 0, n_samples - 1)
            
            windows = traces[np.arange(n_traces)[:, np.newaxis], indices]
            
            # Correlate windows with template
            # template is (pulse_len,)
            # windows is (n_traces, pulse_len)
            
            # We can use our vectorized_correlation helper if we treat template as "predictions"?
            # No, helper expects (n_traces,) predictions vs (n_traces, n_samples).
            # Here we have (n_traces, pulse_len) vs (pulse_len,).
            # We want correlation for each trace.
            
            # Let's write a specific correlation for this.
            # Corr(row, template)
            
            mean_w = np.mean(windows, axis=1, keepdims=True)
            mean_t = np.mean(template)
            
            w_centered = windows - mean_w
            t_centered = template - mean_t
            
            # num: sum(w_c * t_c)
            numerator = np.sum(w_centered * t_centered, axis=1)
            
            ss_w = np.sum(w_centered ** 2, axis=1)
            ss_t = np.sum(t_centered ** 2)
            
            denominator = np.sqrt(ss_w * ss_t)
            
            # Avoid div by zero
            mask = denominator == 0
            denominator[mask] = 1.0
            
            corrs = numerator / denominator
            corrs[mask] = 0.0
            
            all_correlations.append(corrs)
            
        all_correlations = np.array(all_correlations).T # (n_traces, n_rounds)
        
        # Use Otsu's method on the correlations to find the threshold
        # We can compute one threshold for ALL correlations (global) or per round?
        # Global is better if statistics are consistent.
        threshold = self._otsu_threshold(all_correlations.flatten())
        
        predicted_bits = (all_correlations > threshold).astype(np.int8)
        
        # If we are doing "S(n)", we might want to average correlations across traces?
        # Or average traces then correlate?
        # The prompt says "Recover bits using statistical correlation".
        # If we output bits for each trace, we can then vote.
        
        # Let's return the bits for each trace.
        # Wait, the signature I wrote is `-> np.ndarray`.
        # If I return (n_traces, n_rounds), that's fine.
        # If I return (n_rounds,), I need to aggregate.
        # Let's return (n_traces, n_rounds) to be flexible.
        
        return predicted_bits, all_correlations
