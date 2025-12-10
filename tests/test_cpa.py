import pytest
import numpy as np
from ecc_sca_project.src.attacks.cpa import CPAAttack
from ecc_sca_project.src.generation.generator import ECCTraceGenerator

def test_cpa_clean_trace_correlation():
    """
    Verify Correlation = 1.0 when noise=0.
    """
    n_rounds = 10
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=1,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 0.0}
    )
    
    bits = np.ones((1, n_rounds), dtype=int) # All 1s to ensure Add pulses exist
    traces, starts = gen.generate_batch(bits)
    
    cpa = CPAAttack(pulse_len=80, gap=40)
    
    predicted_bits, correlations = cpa.recover_key(traces, starts)
    
    # Check correlations
    # Should be 1.0 for all rounds (since bits are 1 and noise is 0)
    assert np.allclose(correlations, 1.0), f"Correlations should be 1.0, got {correlations}"
    assert np.all(predicted_bits == 1)

def test_cpa_zeros_correlation():
    """
    Verify Correlation is low (or undefined/handled) when bit=0 (no pulse).
    """
    n_rounds = 10
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=1,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 0.0}
    )
    
    bits = np.zeros((1, n_rounds), dtype=int) # All 0s
    traces, starts = gen.generate_batch(bits)
    
    cpa = CPAAttack(pulse_len=80, gap=40)
    
    predicted_bits, correlations = cpa.recover_key(traces, starts)
    
    # If signal is flat 0, correlation with template is undefined (0/0).
    # Our implementation handles this by returning 0.0.
    # Or if there is slight numerical noise, it might be random.
    # But with clean traces, it should be 0.0.
    
    # Note: The generator might output exactly 0.0.
    # Let's check.
    
    assert np.all(predicted_bits == 0)
    # Correlations should be 0 or NaN (handled as 0)
    assert np.allclose(correlations, 0.0)

def test_cpa_noisy_performance():
    """
    Verify CPA works with noise.
    """
    n_rounds = 50
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=5,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 0.5}
    )
    
    bits = np.random.randint(0, 2, size=(5, n_rounds))
    traces, starts = gen.generate_batch(bits, seed=123)
    
    cpa = CPAAttack(pulse_len=80, gap=40)
    
    predicted_bits, correlations = cpa.recover_key(traces, starts)
    
    accuracy = np.mean(predicted_bits == bits)
    assert accuracy > 0.9, f"CPA accuracy too low on noisy traces: {accuracy}"
