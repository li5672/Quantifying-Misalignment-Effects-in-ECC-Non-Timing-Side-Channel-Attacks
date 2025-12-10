# Quantifying Misalignment Effects in ECC Non-Timing Side-Channel Attacks

This project provides a comprehensive benchmark for evaluating the robustness of Side-Channel Analysis (SCA) attacks against temporal jitter in Elliptic Curve Cryptography (ECC). It implements a complete pipeline from synthetic trace generation to advanced attacks and rigorous scientific metrics.

## 🚀 Features

- **Vectorized Trace Generation**: High-performance generation of synthetic ECC traces with configurable "Double-and-Add" leakage, colored noise, and jitter.
- **Attack Suite**:
    - **SPA (Simple Power Analysis)**: Automated baseline using Otsu's adaptive thresholding.
    - **CPA (Correlation Power Analysis)**: Statistical attack using Matched Filters.
    - **Deep Learning (1D-CNN)**: State-of-the-art attack robust to jitter (Translation Invariance).
- **Preprocessing**: Cross-Correlation based trace re-alignment.
- **Scientific Metrics**:
    - **$S(n)$**: Success rate vs. trace count.
    - **$N_{80}$**: Trace budget for 80% success.
    - **$r(j)$**: Jitter Degradation Factor.

## 🛠️ Installation

Prerequisites: Python 3.9+

1.  **Clone the repository** (or navigate to the project folder):
    ```bash
    cd ecc_sca_project
    ```

2.  **Install Dependencies** (using pip):
    ```bash
    # Create a virtual environment (optional but recommended)
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install packages
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

All scripts should be run from the `ecc_sca_project` directory with `PYTHONPATH=..` (so Python can see the project package).

### 1. Reproducing the Report Figures

We provide dedicated scripts to reproduce every figure in the final report. This is the primary way to verify our results.

**Step 1: Visualizing the Attack (Figures 4 & 5)**
First, generate the qualitative plots that demonstrate the attack principles and the effect of alignment.
```bash
# Figure 4a: SPA Baseline Trace (J=0)
PYTHONPATH=.. python3 demo_spa.py --jitter 0 --noise 0.2

# Figure 4b: CPA Correlation Peaks
PYTHONPATH=.. python3 demo_cpa.py --traces 50 --noise 1.0

# Figure 5: Alignment Demo (J=50)
PYTHONPATH=.. python3 demo_alignment.py --jitter 50 --pulse_len 20
```
*Outputs: `results/spa_demo_j0.png`, `results/cpa_demo.png`, `results/alignment_demo_j50_p20.png`*

**Step 2: Benchmarking Performance (Figures 2 & 3)**
Run the full statistical benchmark to generate the success rate curves and jitter degradation plot.
```bash
PYTHONPATH=.. python3 demo_full_benchmark.py --noise 3.0
```
*Outputs: `results/benchmark_success_curves.png`, `results/benchmark_jitter_degradation.png`*

**Step 3: Final Method Comparison (Figure 1)**
Generate the comprehensive comparison between SPA, CPA, and CNN under high noise and jitter.
```bash
PYTHONPATH=.. python3 benchmark_comparison.py --jitter 20 --noise 3.0
```
*Output: `results/final_comparison_all_methods.png`*

### 2. Additional Analyses

**CPA Success Rate Analysis**
Evaluate CPA performance across different jitter levels.
```bash
PYTHONPATH=.. python3 benchmark_cpa.py --noise 3.0
```
*Output: `results/cpa_success_curves.png`*

## 📂 Project Structure

```
ecc_sca_project/
├── src/
│   ├── generation/  # Trace generation logic (generator.py, noise.py)
│   ├── attacks/     # Attack implementations (spa.py, cpa.py, cnn.py)
│   ├── metrics/     # Statistics and metrics (calculator.py, statistics.py)
│   └── utils/       # Helpers (alignment.py, visualization.py)
├── tests/           # Pytest suite
├── config/          # Hydra configuration
├── results/         # Output plots and results
├── benchmark_*.py   # Scientific benchmark scripts
└── demo_*.py        # Demonstration scripts
```

## 🧪 Running Tests

To verify the correctness of all components:
```bash
PYTHONPATH=.. python3 -m pytest tests/
```
