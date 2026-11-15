# HybridTailNet: Solving Information Bottlenecks via Topological Metadata Tails

[![DOI](https://zenodo.org/badge/1458637375.svg)](https://doi.org/10.5281/zenodo.24243846)


Standard Multi-Layer Perceptrons (MLPs) treat every signal as a flat, memoryless scalar. When a network compresses data through narrow **bottleneck layers**, downstream neurons lose all structural context—they cannot tell whether a `0.0` came from an inactive sensor, missing data (`NaN`), or the sum of 100 active pathways.

**HybridTailNet** introduces a new dual-node architecture where signals carry their own topological history ("tails") across layers, and specialized **Non-Neurons** periodically compress this history into context-aware decisions.

---

## ⚙️ Core Mechanics: How Signals Evolve

Instead of transmitting a single floating-point number $v$, every connection in HybridTailNet transmits a **dynamic state vector** consisting of a primary activation value followed by a growing **metadata tail**:

$$\mathbf{s} = [v, \; t_1, \; t_2, \; \dots, \; t_k]$$

### 1. Initialization (Input Layer)

Every active, valid input feature $x_i$ enters the network initialized with a depth marker of `1.0`:

* **Active input ($x_i = 2.0$):** $\rightarrow [2.0, \; 1.0]$
* **Missing / Inactive input (`NaN`):** $\rightarrow [0.0, \; 0.0]$ *(carries zero structural weight)*

### 2. Tail Construction & Growth (Standard Neuron Layers)

Standard neurons act as **local accumulators**. When a neuron in layer $L$ receives $N$ incoming vectors $\{\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(N)}\}$:

1. **Primary Value:** It computes a standard weighted/mean transformation $v_{\text{new}}$ from the incoming primary values $v^{(i)}$.
2. **Tail Summation:** It sums the incoming metadata tails element-wise across all $N$ connections: $T_{\text{sum}} = \sum_{i=1}^{N} \mathbf{t}^{(i)}$.
3. **Depth Appending:** It shifts the summed tail to the right and appends a new marker `1.0` at the end to record the current layer step.

### 3. Tail Collapse & Renormalization ("Non-Neuron" Bottlenecks)

Placed inside narrow **bottleneck layers**, **Non-Neurons** act as **global context compressors**. Unlike standard neurons that only look at primary values, a Non-Neuron reads the entire vector—both the signal and its topological history:

1. **Zero-Padding:** Because tails grow dynamically with depth, the incoming vector $[v, \; t_1, \dots, t_k]$ is zero-padded to a fixed buffer size (e.g., `max_len = 10`). Tail counters are scaled (e.g., $\times 0.1$) for numerical stability.
2. **Shared Micro-Network Inference:** A single, lightweight micro-network (shared across **all** Non-Neurons in the entire architecture) processes the full 10-dimensional vector. This allows the Non-Neuron to modulate its output based on *how many paths* and *how many layers* generated the signal.
3. **Tail Reset (Collapse):** Once the Non-Neuron outputs its context-aware scalar $v_{\text{out}}$, it **collapses (resets) the accumulated tail** back to a fresh unit marker:

$$\mathbf{s}_{\text{collapsed}} = [v_{\text{out}}, \; 1.0]$$

---

## 🔍 Step-by-Step Numerical Trace

Suppose an active signal $x = 2.0$ enters a network where wide layers have **10 standard neurons**, followed by a **Non-Neuron bottleneck**:

* **Step 0 (Input):**
  Active input arrives with a unit marker:
  $$\mathbf{s}_0 = [2.0, \; 1.0]$$

* **Step 1 (Layer 1 — Standard Neuron):**
  Transforms the primary value to `3.1` and appends a new depth marker `1.0`:
  $$\mathbf{s}_1 = [3.1, \; 1.0]$$

* **Step 2 (Layer 2 — Standard Neuron receiving from 10 Layer-1 neurons):**
  * Primary values aggregate to `0.17`.
  * The 10 tails (`[1.0]`) are summed: $1.0 \times 10 = 10.0$.
  * A new depth marker `1.0` is appended:
    $$\mathbf{s}_2 = [0.17, \; 10.0, \; 1.0]$$
    *(The tail now encodes that 10 pathways converged across 2 layers.)*

* **Step 3 (Layer 3 — Standard Neuron receiving from 10 Layer-2 neurons):**
  * Primary values aggregate to `0.24`.
  * Tails (`[10.0, 1.0]`) are summed across 10 incoming paths: $[100.0, \; 10.0]$.
  * A new depth marker `1.0` is appended:
    $$\mathbf{s}_3 = [0.24, \; 100.0, \; 10.0, \; 1.0]$$

* **Step 4 (Layer 4 — Non-Neuron Bottleneck):**
  * **Zero-Padding:** $\mathbf{s}_3$ is padded to length 10:
    $$[0.24, \; 100.0, \; 10.0, \; 1.0, \; 0.0, \; 0.0, \; 0.0, \; 0.0, \; 0.0, \; 0.0]$$
  * **Context Inference:** The shared micro-network reads both the activation (`0.24`) and its path multiplicity (`100, 10, 1`), producing `0.51`.
  * **Collapse:** The tail is wiped clean and reset to `1.0`:
    $$\mathbf{s}_4 = [0.51, \; 1.0]$$

---

## 📊 Experimental Results

| Benchmark Task | Metric | HybridTailNet (Ours) | Standard MLP (Baseline) |
| :--- | :---: | :---: | :---: |
| **3-Bottleneck Structural Logic** *(20 train samples)* | Test MSE ↓ | **0.3340** | 0.5657 |
| **3-Bottleneck Structural Logic** *(20 train samples)* | Accuracy ↑ | **70.0%** | 58.5% |
| **Spaceship Titanic** *(1,778 vs 2,129 params)* | Val LogLoss ↓ | **0.4348** | 0.4408 |

---

## 🚀 Why This Matters

* **Zero Information Loss in Bottlenecks:** Even if `0.24` approaches `0.0`, the Non-Neuron sees `[100, 10, 1]` and knows high-volume activity occurred upstream.
* **Native Missing-Data Handling:** In tabular tasks (like *Spaceship Titanic*), missing values (`NaN`) simply omit the `.1` marker. The tail naturally tracks exact feature completeness per sample without extra imputation flags.
* **Parameter Efficiency & Weight Sharing:** Because every Non-Neuron shares the same micro-network, structural compression rules learned in one bottleneck immediately transfer to all other bottlenecks in the model.

---

## 🛠️ Quick Start
---

## 📜 Citation

If you use **HybridTailNet** or dynamic topological metadata tails in your research, please cite this repository:

```bibtex
@software{hybridtailnet_2026,
  title        = {HybridTailNet: Solving Information Bottlenecks via Topological Metadata Tails},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24243846},
  url          = {https://doi.org/10.5281/zenodo.24243846}
}
```


```bash
pip install -r requirements.txt
python benchmark_titanic.py
```
