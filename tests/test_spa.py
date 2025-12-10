import pytest
import numpy as np
from ecc_sca_project.src.attacks.spa import SPAAttack
from ecc_sca_project.src.generation.generator import ECCTraceGenerator

def test_spa_clean_trace():
    """
    Verify SPA recovers 100% of bits on a clean trace (no noise, no jitter).
    """
    n_rounds = 50
    gen = ECCTraceGenerator(
        trace_length=1000, # Dummy
        n_traces=1,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 0.0}
    )
    
    bits = np.random.randint(0, 2, size=(1, n_rounds))
    traces, starts = gen.generate_batch(bits)
    
    # Pulse len and gap must match generator
    # Generator uses pulse_len=80, gap=40
    spa = SPAAttack(pulse_len=80, gap=40)
    
    predicted_bits, threshold = spa.recover_key(traces[0], starts[0])
    
    # Check accuracy
    accuracy = np.mean(predicted_bits == bits[0])
    assert accuracy == 1.0, f"SPA failed on clean trace. Accuracy: {accuracy}"
    
    # Check threshold is reasonable
    # Energy of 0 (Add missing) should be 0.
    # Energy of 1 (Add present) should be > 0.
    # Threshold should be between them.
    assert threshold > 0.0

def test_spa_noisy_trace():
    """
    Verify SPA works reasonably well on a noisy trace.
    """
    n_rounds = 100
    # Moderate noise
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=1,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 0.5} # Signal amplitude is 1.0
    )
    
    bits = np.random.randint(0, 2, size=(1, n_rounds))
    traces, starts = gen.generate_batch(bits, seed=123)
    
    spa = SPAAttack(pulse_len=80, gap=40)
    
    predicted_bits, threshold = spa.recover_key(traces[0], starts[0])
    
    accuracy = np.mean(predicted_bits == bits[0])
    # With noise 0.5 and signal 1.0, separation should still be decent but not perfect?
    # Energy integration improves SNR.
    # sqrt(sum(noise^2)) vs sum(signal^2)
    # Let's expect > 90% accuracy
    assert accuracy > 0.9, f"SPA accuracy too low on noisy trace: {accuracy}"

def test_otsu_thresholding_logic():
    """
    Test Otsu's method on a synthetic bimodal distribution.
    """
    spa = SPAAttack(pulse_len=10, gap=10)
    
    # Create bimodal distribution
    # Class 0: mean 10, std 1
    # Class 1: mean 20, std 1
    c0 = np.random.normal(10, 1, 1000)
    c1 = np.random.normal(20, 1, 1000)
    values = np.concatenate([c0, c1])
    
    threshold = spa._otsu_threshold(values)
    
    # Threshold should be between the two distributions (10+3*1=13 and 20-3*1=17)
    # With averaging, it should be closer to 15.
    assert 13.0 < threshold < 17.0, f"Otsu threshold {threshold} far from expected ~15.0"
