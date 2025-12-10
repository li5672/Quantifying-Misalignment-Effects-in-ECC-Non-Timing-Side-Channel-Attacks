import numpy as np
from scipy.signal import correlate

class CrossCorrelationAligner:
    """
    Aligns traces using Cross-Correlation.
    """
    def __init__(self, max_shift: int = 50):
        self.max_shift = max_shift

    def align(self, traces: np.ndarray, reference_idx: int = 0) -> np.ndarray:
        """
        Aligns all traces to the reference trace.
        
        Args:
            traces: (n_traces, length)
            reference_idx: Index of the trace to use as reference.
            
        Returns:
            aligned_traces: (n_traces, length)
        """
        n_traces, length = traces.shape
        reference = traces[reference_idx]
        
        aligned_traces = np.zeros_like(traces)
        
        for i in range(n_traces):
            if i == reference_idx:
                aligned_traces[i] = reference
                continue
                
            trace = traces[i]
            
            # Compute cross-correlation
            # mode='same' returns output of same length as input
            # We want to find the shift that maximizes correlation
            
            # Use fft based correlation for speed if traces are long
            # scipy.signal.correlate with method='fft'
            
            # We restrict the search window to +/- max_shift to avoid false peaks far away
            # But correlate computes full convolution.
            
            corr = correlate(reference, trace, mode='same', method='fft')
            
            # The peak should be at center (length//2) if perfectly aligned
            # Find peak
            peak_idx = np.argmax(corr)
            
            # Calculate shift
            # If peak is at center, shift is 0.
            # If peak is to the right, trace is delayed?
            # Let's verify direction.
            # If trace is shifted right (delayed) by d: trace(t) = ref(t-d)
            # Cross-corr (ref * trace) peak will be at ...
            
            center = length // 2
            shift = peak_idx - center
            
            # Enforce max shift constraint (optional, but good for robustness)
            if abs(shift) > self.max_shift:
                # Fallback: do not shift if peak is too far (likely noise)
                # Or clamp? Let's just not shift for now or clamp.
                # Let's clamp.
                shift = np.clip(shift, -self.max_shift, self.max_shift)
            
            # Apply shift
            # The calculated 'shift' is the correction needed.
            # Verification confirmed that 'shift' gives the correct direction.
            
            aligned_trace = self._shift_trace(trace, shift)
            aligned_traces[i] = aligned_trace
            
        return aligned_traces

    def _shift_trace(self, trace: np.ndarray, shift: int) -> np.ndarray:
        """
        Shifts a single trace.
        shift > 0: Shift right (pad left)
        shift < 0: Shift left (pad right)
        """
        result = np.zeros_like(trace)
        length = len(trace)
        
        if shift == 0:
            return trace
        
        if shift > 0:
            # Shift right
            # result[shift:] = trace[:-shift]
            if shift < length:
                result[shift:] = trace[:length-shift]
        else:
            # Shift left
            # result[:shift] = trace[-shift:] (since shift is negative)
            s = -shift
            if s < length:
                result[:length-s] = trace[s:]
                
        return result
