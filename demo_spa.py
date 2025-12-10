import numpy as np
import matplotlib.pyplot as plt
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.spa import SPAAttack
import os

import argparse

def run_demo(max_jitter=0, noise_level=0.2, n_rounds=20, gap=40):
    print(f"Running SPA Demo with Jitter={max_jitter}, Noise={noise_level}, KeyLen={n_rounds}, Gap={gap}...")
    
    # 1. Setup Parameters
    n_traces = 5
    pulse_len = 80
    # gap passed as arg
    
    # 2. Generate Traces
    print("Generating synthetic traces...")
    gen = ECCTraceGenerator(
        trace_length=1000, 
        n_traces=n_traces,
        tailroom=100,
        max_jitter=max_jitter,
        noise_params={'level': noise_level, 'color_factor': 0.5, 'baseline_drift': 0.05}
    )
    
    bits = np.random.randint(0, 2, size=(n_traces, n_rounds))
    traces, starts = gen.generate_batch(bits, seed=42)
    
    # 3. Run SPA Attack
    print("Running SPA Attack...")
    
    round_len = pulse_len * 2 + gap
    base_pos = np.arange(n_rounds) * round_len + 10
    
    nominal_starts = np.tile(base_pos, (n_traces, 1))
    
    # Attack the first trace
    target_trace_idx = 0
    
    # Use nominal starts for the attack to simulate misalignment
    predicted_bits, threshold = spa_attack_wrapper(traces[target_trace_idx], nominal_starts[target_trace_idx], pulse_len, gap)
    
    actual_bits = bits[target_trace_idx]
    
    accuracy = np.mean(predicted_bits == actual_bits)
    print(f"Trace {target_trace_idx} Analysis:")
    print(f"Actual Bits:    {actual_bits[:10]}...")
    print(f"Predicted Bits: {predicted_bits[:10]}...")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Otsu Threshold: {threshold:.4f}")
    
    # 4. Visualize
    print("Generating visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot a segment
    plot_len = min(len(traces[0]), round_len * 3 + max_jitter * 2)
    
    t = np.arange(plot_len)
    plt.plot(t, traces[target_trace_idx, :plot_len], label='Power Trace', alpha=0.8)
    
    # Mark "Add" windows based on NOMINAL starts (where we looked)
    for r in range(min(3, n_rounds)):
        s = nominal_starts[target_trace_idx, r]
        add_start = s + pulse_len + gap
        add_end = add_start + pulse_len
        
        # Highlight the window we CHECKED
        plt.axvspan(add_start, add_end, color='red', alpha=0.1, label='Integration Window (Nominal)' if r==0 else "")
        
        # Annotate bit
        plt.text(s, np.max(traces[target_trace_idx])*1.1, f"Bit: {actual_bits[r]}", fontsize=10)

    plt.title(f"SPA Demo (Jitter={max_jitter}): Accuracy {accuracy*100:.0f}%")
    plt.xlabel("Time (samples)")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join("results", f"spa_demo_j{max_jitter}.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {os.path.abspath(output_path)}")

def spa_attack_wrapper(trace, starts, pulse_len, gap):
    spa = SPAAttack(pulse_len=pulse_len, gap=gap)
    return spa.recover_key(trace, starts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=int, default=0, help="Max jitter amount")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level")
    parser.add_argument("--key_len", type=int, default=20, help="Key length (number of rounds)")
    parser.add_argument("--gap", type=int, default=40, help="Gap between Double and Add pulses")
    args = parser.parse_args()
    
    run_demo(max_jitter=args.jitter, noise_level=args.noise, n_rounds=args.key_len, gap=args.gap)
