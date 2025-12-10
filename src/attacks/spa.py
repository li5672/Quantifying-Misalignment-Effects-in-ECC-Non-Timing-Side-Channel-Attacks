import numpy as np
from typing import Tuple, Optional

class SPAAttack:
    """
    Automated Simple Power Analysis (SPA) using Otsu's Method for adaptive thresholding.
    """
    def __init__(self, pulse_len: int, gap: int):
        self.pulse_len = pulse_len
        self.gap = gap

    def extract_features(self, trace: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """
        Extracts energy of the 'Add' window for each round.
        
        Args:
            trace: 1D array (single trace).
            starts: 1D array of start indices for each round.
            
        Returns:
            features: 1D array of energy values for the 'Add' windows.
        """
        features = []
        for s in starts:
            # 'Add' pulse is expected at s + pulse_len + gap
            # We want to integrate energy in that window
            start_add = s + self.pulse_len + self.gap
            end_add = start_add + self.pulse_len
            
            # Handle boundary checks
            if start_add >= len(trace):
                features.append(0.0)
                continue
                
            w_add = trace[start_add : min(end_add, len(trace))]
            
            # Energy = sum(x^2)
            energy = np.sum(w_add ** 2)
            features.append(energy)
            
        return np.array(features)

    def _otsu_threshold(self, values: np.ndarray) -> float:
        """
        Computes the optimal threshold using Otsu's Method.
        Minimizes intra-class variance (equivalent to maximizing inter-class variance).
        """
        # Create a histogram
        # We need a reasonable number of bins.
        # For continuous values, we can just sort them and test split points?
        # Or bin them. Binning is faster.
        
        hist, bin_edges = np.histogram(values, bins=256, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Probabilities and means
        # weight_background (w0) and weight_foreground (w1)
        # cumsum of hist
        
        # Normalize hist to sum to 1 (it's density=True, but let's be safe with discrete sum)
        hist = hist / np.sum(hist)
        
        cum_sum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * bin_centers)
        
        global_mean = cum_mean[-1]
        
        # Inter-class variance: sigma_b^2 = w0 * w1 * (mu0 - mu1)^2
        # Simplified: sigma_b^2 = (global_mean * w0 - mu_k)^2 / (w0 * (1-w0))
        # where mu_k is cumulative mean up to k
        
        # Avoid division by zero
        valid_mask = (cum_sum > 0) & (cum_sum < 1)
        
        # Compute variance for all valid splits
        # numerator = (global_mean * w0 - mu_k)^2
        numerator = (global_mean * cum_sum[valid_mask] - cum_mean[valid_mask]) ** 2
        denominator = cum_sum[valid_mask] * (1.0 - cum_sum[valid_mask])
        
        variance = numerator / denominator
        
        # Find indices of max variance (handle plateaus)
        max_var = np.max(variance)
        # Use a small epsilon for floating point comparison
        optimal_indices = np.where(variance >= max_var * 0.9999)[0]
        
        # Map back to valid_mask indices
        # We need to find the corresponding bin centers.
        # valid_mask has the same length as variance? No.
        # variance was computed only for valid_mask.
        # So optimal_indices are indices INTO variance array.
        # We need to map them to indices into the original arrays (cum_sum, etc).
        
        # Let's get the indices of valid_mask where it is True
        valid_indices_in_original = np.where(valid_mask)[0]
        
        # Now map optimal_indices to original indices
        best_original_indices = valid_indices_in_original[optimal_indices]
        
        # Average the bin centers corresponding to these indices
        best_threshold = np.mean(bin_centers[best_original_indices])
        
        return best_threshold

    def recover_key(self, trace: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Recovers the key (bits) from a single trace.
        
        Args:
            trace: 1D array.
            starts: Start indices of rounds.
            
        Returns:
            bits: Predicted bits (0 or 1).
            threshold: The calculated Otsu threshold.
        """
        # 1. Extract features (energies of Add windows)
        energies = self.extract_features(trace, starts)
        
        # 2. Compute Otsu Threshold
        threshold = self._otsu_threshold(energies)
        
        # 3. Classify
        bits = (energies > threshold).astype(np.int8)
        
        return bits, threshold
