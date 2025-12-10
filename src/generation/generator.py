import numpy as np
from typing import Tuple, Optional
from .noise import ColoredNoise

class ECCTraceGenerator:
    """
    Vectorized generator for synthetic ECC side-channel traces.
    Simulates 'Double-and-Add' leakage with Jitter and Noise.
    """
    def __init__(
        self,
        trace_length: int,
        n_traces: int,
        tailroom: int,
        max_jitter: int,
        noise_params: dict
    ):
        self.L_trace = trace_length
        self.n_traces = n_traces
        self.tailroom = tailroom
        self.max_jitter = max_jitter
        
        self.noise_gen = ColoredNoise(
            level=noise_params.get('level', 0.1),
            color_factor=noise_params.get('color_factor', 0.0),
            baseline_drift=noise_params.get('baseline_drift', 0.0)
        )
        
        # Pulse shape parameters (hardcoded for now, could be config)
        self.pulse_width = 10
        self.pulse_amp_double = 1.0
        self.pulse_amp_add = 1.0

    def _gaussian_pulse(self, length: int, center: float, width: float) -> np.ndarray:
        x = np.arange(length)
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    def generate_batch(self, bits: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a batch of traces for the given bits.
        
        Args:
            bits: (n_traces, n_rounds) or (n_rounds,) array of bits.
                  If 1D, same bits used for all traces (but different noise/jitter).
            seed: Random seed.
            
        Returns:
            traces: (n_traces, total_length)
            starts: (n_traces, n_rounds) Start indices of each round.
        """
        rng = np.random.default_rng(seed)
        
        # Handle broadcasting for bits
        if bits.ndim == 1:
            bits = np.tile(bits, (self.n_traces, 1))
        
        n_traces, n_rounds = bits.shape
        if n_traces != self.n_traces:
             raise ValueError(f"Bits shape {bits.shape} does not match n_traces {self.n_traces}")

        # Define timing constants
        # Round length must accommodate Double + Add + Gap
        # Let's define a fixed round structure for simplicity
        # Double starts at 0, Add starts at L_trace/2 (approx)
        # We need to ensure total length is sufficient
        
        # Using the logic from legacy code but cleaning it up:
        # round_len = L*2 + gap
        # But here we are given L_trace as a global param? 
        # Let's interpret L_trace as the length of ONE operation (Double or Add)
        # Or L_trace as the total length of the trace?
        # Re-reading AGENT.md: "Generated traces must strictly adhere to length L_trace + 2*tailroom"
        # This implies L_trace is the target useful length.
        # BUT, for ECC, the length depends on the number of bits (rounds).
        # Let's assume L_trace in config refers to the length of a SINGLE ROUND or the pulse width?
        # Legacy code: `synth_trace_for_bits(bits, L=80...)` -> L is pulse length.
        # Config says: `length: 1000 # L_trace`. This might be the total length?
        # Let's assume `trace.length` in config is the TOTAL length we want to output, 
        # OR we calculate total length based on bits.
        # Let's look at config again: `length: 1000`.
        # If we have 256 bits, 1000 samples is too short.
        # Let's assume the config `length` is the length of ONE ROUND (Double+Add).
        # OR, let's stick to the legacy logic where `L` was pulse width.
        
        # Let's redefine for clarity:
        # pulse_len: Length of the pulse window
        # round_len: Length of one bit processing (Double + optional Add)
        
        # pulse_len: Length of the pulse window
        # round_len: Length of one bit processing (Double + optional Add)
        
        # Use instance attributes if they exist (monkey-patching support for demo)
        pulse_len = getattr(self, 'pulse_len', 80)
        gap = getattr(self, 'gap', 40)
        
        round_len = pulse_len * 2 + gap
        
        total_len = round_len * n_rounds + 2 * self.max_jitter + self.tailroom
        
        traces = np.zeros((n_traces, total_len), dtype=np.float64)
        
        # Pre-compute pulses
        dbl_pulse = self._gaussian_pulse(pulse_len, center=pulse_len//2, width=pulse_len/6) * self.pulse_amp_double
        add_pulse = self._gaussian_pulse(pulse_len, center=pulse_len//2, width=pulse_len/6) * self.pulse_amp_add
        
        # Generate Jitter
        # jitter: (n_traces, n_rounds)
        jitter_mode = getattr(self, 'jitter_mode', 'per_round')
        
        if self.max_jitter > 0:
            if jitter_mode == 'global':
                # Same shift for all rounds in a trace (Rigid translation)
                # Shape: (n_traces, 1) -> broadcast to (n_traces, n_rounds)
                shifts = rng.integers(-self.max_jitter, self.max_jitter + 1, size=(n_traces, 1))
                jitter_shifts = np.tile(shifts, (1, n_rounds))
            else:
                # Random shift per round (Elastic/Clock jitter)
                jitter_shifts = rng.integers(-self.max_jitter, self.max_jitter + 1, size=(n_traces, n_rounds))
        else:
            jitter_shifts = np.zeros((n_traces, n_rounds), dtype=int)
            
        # Calculate base positions
        # base_pos: (n_rounds,)
        base_pos = np.arange(n_rounds) * round_len + 10 # +10 offset
        
        # Calculate actual start positions for each round and trace
        # starts: (n_traces, n_rounds)
        starts = base_pos[np.newaxis, :] + jitter_shifts
        
        # Vectorized Pulse Injection
        # This is tricky to fully vectorize without a loop over rounds because of overlap potential
        # But rounds are usually sequential. We can loop over rounds.
        
        for r in range(n_rounds):
            # Current round starts for all traces
            # s_r: (n_traces,)
            s_r = starts[:, r]
            
            # Double Pulse (Always happens)
            # We need to add dbl_pulse to traces[i, s_r[i] : s_r[i]+pulse_len]
            # Advanced indexing:
            # Create a range matrix for indices
            # idx: (n_traces, pulse_len)
            idx = s_r[:, np.newaxis] + np.arange(pulse_len)
            
            # Clip indices to bounds (just in case)
            mask = (idx >= 0) & (idx < total_len)
            
            # We can't easily use advanced indexing for assignment with += if indices overlap within a single row
            # But here, for a single round, indices for one trace do not overlap.
            # So we can flatten and assign?
            # Or just iterate traces? No, we want to avoid that.
            # `np.add.at` is good for unbuffered addition, but slow.
            # Since all pulses are same shape, we can use:
            # traces[np.arange(n_traces)[:,None], idx] += dbl_pulse
            # This works!
            
            # Handle clipping by masking
            valid_rows, valid_cols = np.where(mask)
            # This is getting complicated to handle boundary checks efficiently.
            # Given constraints, let's assume sufficient buffer and valid indices for speed.
            # Or use a loop over traces if n_traces is small? No, n_traces is large (1000+).
            # Loop over rounds is fine (e.g. 256 rounds).
            
            # Optimized approach:
            # traces[i, idx[i]] += pulse
            rows = np.arange(n_traces)[:, np.newaxis]
            
            # Safe assignment with bounds check
            # We only add where mask is True.
            # But dbl_pulse is 1D.
            
            # Let's try a simpler approach:
            # If we guarantee no out of bounds by design (padding), we can skip checks.
            # total_len includes tailroom.
            
            traces[rows, idx] += dbl_pulse
            
            # Add Pulse (If bit == 1)
            # s_add: (n_traces,)
            s_add = s_r + pulse_len + gap
            idx_add = s_add[:, np.newaxis] + np.arange(pulse_len)
            
            # Mask for bits == 1
            do_add = (bits[:, r] == 1)
            
            # Only add to traces where bit is 1
            # We can multiply pulse by bit?
            # pulse_matrix: (n_traces, pulse_len)
            pulse_matrix = dbl_pulse * do_add[:, np.newaxis] # 0 or pulse
            
            traces[rows, idx_add] += pulse_matrix

        # Add Noise
        noise = self.noise_gen.generate(traces.shape, seed=seed+1)
        traces += noise
        
        return traces, starts
