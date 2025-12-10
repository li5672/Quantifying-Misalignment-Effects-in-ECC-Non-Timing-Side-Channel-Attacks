import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.cpa import CPAAttack

def run_demo(n_traces=50, noise_level=1.0, n_rounds=20, gap=40):
    print(f"Running CPA Demo with N={n_traces}, Noise={noise_level}, KeyLen={n_rounds}, Gap={gap}...")
    
    # 1. Setup Parameters
    pulse_len = 80
    # gap passed as arg
    
    # 2. Generate Traces
    print("Generating synthetic traces...")
    gen = ECCTraceGenerator(
        trace_length=1000, 
        n_traces=n_traces,
        tailroom=100,
        max_jitter=0, # CPA (Matched Filter) is sensitive to jitter without alignment
        noise_params={'level': noise_level, 'color_factor': 0.5, 'baseline_drift': 0.05}
    )
    
    bits = np.random.randint(0, 2, size=(n_traces, n_rounds))
    traces, starts = gen.generate_batch(bits, seed=42)
    
    # 3. Run CPA Attack
    print("Running CPA Attack...")
    cpa = CPAAttack(pulse_len=pulse_len, gap=gap)
    
    # We attack all traces
    predicted_bits, correlations = cpa.recover_key(traces, starts)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_bits == bits)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # 4. Visualize Correlation for one trace
    print("Generating visualization...")
    target_trace_idx = 0
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Power Trace
    plt.subplot(2, 1, 1)
    round_len = pulse_len * 2 + gap
    plot_len = min(len(traces[0]), round_len * 5) # Show 5 rounds
    t = np.arange(plot_len)
    plt.plot(t, traces[target_trace_idx, :plot_len], label='Power Trace', alpha=0.7)
    plt.title(f"Trace Segment (First 5 Rounds) - Bit: {bits[target_trace_idx, 0]}")
    plt.ylabel("Power")
    plt.legend()
    
    # Subplot 2: Correlation with Template (for the same trace)
    # We need to reconstruct the full correlation trace or just plot the peak values?
    # The CPAAttack returns 'correlations' which are the PEAK correlations for each round.
    # To visualize the "Correlation Trace" (sliding window), we would need to modify the attack 
    # or just do it here manually for the demo.
    
    # Let's plot the Peak Correlations for the first 20 rounds
    plt.subplot(2, 1, 2)
    rounds = np.arange(n_rounds)
    corr_values = correlations[target_trace_idx]
    actual_b = bits[target_trace_idx]
    
    # Color bars by actual bit
    colors = ['blue' if b == 0 else 'red' for b in actual_b]
    
    plt.bar(rounds, corr_values, color=colors, alpha=0.7)
    plt.axhline(y=0.5, color='k', linestyle='--', label='Threshold (0.5)')
    
    # Add labels
    for r, v in enumerate(corr_values):
        plt.text(r, v + 0.05, f"{actual_b[r]}", ha='center', fontsize=8)
        
    plt.title(f"Correlation Peaks per Round (Red=Bit 1, Blue=Bit 0) - Accuracy: {accuracy*100:.1f}%")
    plt.xlabel("Round Index")
    plt.ylabel("Pearson Correlation")
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join("results", "cpa_demo.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=50, help="Number of traces")
    parser.add_argument("--noise", type=float, default=1.0, help="Noise level")
    parser.add_argument("--key_len", type=int, default=20, help="Key length (number of rounds)")
    parser.add_argument("--gap", type=int, default=40, help="Gap between Double and Add pulses")
    args = parser.parse_args()
    
    run_demo(n_traces=args.traces, noise_level=args.noise, n_rounds=args.key_len, gap=args.gap)
