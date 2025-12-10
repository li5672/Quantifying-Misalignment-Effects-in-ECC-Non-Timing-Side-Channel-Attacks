import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.utils.alignment import CrossCorrelationAligner

def run_demo(n_traces=50, max_jitter=30, noise_level=0.5, pulse_len=80):
    print(f"Running Alignment Demo with N={n_traces}, Jitter={max_jitter}, Noise={noise_level}, PulseLen={pulse_len}...")
    
    # 1. Generate Traces
    print("Generating synthetic traces...")
    # Note: We need to pass pulse_len to the generator, but ECCTraceGenerator might not take it in __init__?
    # Let's check generator.py. It seems it might be hardcoded or passed in config.
    # Actually, looking at generator.py (I can't see it right now, but I recall), it might default to 80.
    # Let's assume we need to modify generator.py or just subclass/monkeypatch if it's not an arg.
    # Wait, I implemented generator.py. Let me check it first.
    
    gen = ECCTraceGenerator(
        trace_length=1000, 
        n_traces=n_traces,
        tailroom=100,
        max_jitter=max_jitter,
        noise_params={'level': noise_level, 'color_factor': 0.1, 'baseline_drift': 0.01}
    )
    # Hack: Override pulse_len if the class allows it, or if I need to pass it.
    # If ECCTraceGenerator doesn't accept pulse_len, I might need to update it.
    # Let's try setting it directly after init if it's an attribute.
    gen.pulse_len = pulse_len
    gen.gap = 100 # Increase gap to avoid cycle skipping during alignment
    gen.jitter_mode = 'global' # Use rigid jitter so simple alignment works
    
    # Use the SAME bit sequence for all traces to clearly show alignment
    # If bits are different, traces will look different even if aligned.
    bits_single = np.random.randint(0, 2, size=(1, 20))
    bits = np.tile(bits_single, (n_traces, 1))
    
    traces, starts = gen.generate_batch(bits, seed=42)
    
    # 2. Align Traces
    print("Aligning traces...")
    # We use the first trace as reference (assuming we found a good one, or use mean)
    # In a real scenario, we'd use a clean template. Here we use trace[0] which has noise/jitter.
    # But wait, trace[0] also has jitter. 
    # The aligner needs a "template". 
    # Let's use a synthetic template for perfect alignment demonstration?
    # Or just use trace[0] as the "anchor".
    
    aligner = CrossCorrelationAligner(max_shift=max_jitter*2)
    aligned_traces = aligner.align(traces, reference_idx=0)
    
    # 3. Visualize
    print("Generating visualization...")
    plt.figure(figsize=(12, 10))
    
    # Colors for individual traces
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # Plot Before Alignment
    plt.subplot(2, 1, 1)
    mean_before = np.mean(traces, axis=0)
    
    zoom_start = 0
    zoom_end = 400
    
    # Plot 10 individual traces as requested
    n_plot = min(10, n_traces)
    for i in range(n_plot):
        # Cycle colors
        c = colors[i % len(colors)]
        plt.plot(traces[i], color=c, alpha=0.4, label=f'Trace {i}')
        
    plt.plot(mean_before, color='black', linewidth=3, linestyle='--', label='Mean Trace')
    
    plt.title(f"Before Alignment (Jitter={max_jitter}) - Traces are misaligned")
    # Show legend for all plotted traces
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)
    
    plt.xlim(zoom_start, zoom_end)
    plt.ylabel("Power")
    
    # Plot After Alignment
    plt.subplot(2, 1, 2)
    mean_after = np.mean(aligned_traces, axis=0)
    
    for i in range(n_plot):
        c = colors[i % len(colors)]
        plt.plot(aligned_traces[i], color=c, alpha=0.4, label=f'Trace {i}')
        
    plt.plot(mean_after, color='black', linewidth=3, linestyle='--', label='Mean Trace')
    
    plt.title("After Alignment - Traces snap together")
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)
    
    plt.title("After Alignment - Traces snap together")
    plt.legend(loc='upper right')
    plt.xlim(zoom_start, zoom_end)
    plt.ylabel("Power")
    
    plt.tight_layout()
    output_path = os.path.join("results", f"alignment_demo_j{max_jitter}_p{pulse_len}.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=int, default=50, help="Max jitter amount")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    parser.add_argument("--pulse_len", type=int, default=20, help="Pulse length (width)")
    args = parser.parse_args()
    
    run_demo(max_jitter=args.jitter, noise_level=args.noise, pulse_len=args.pulse_len)
