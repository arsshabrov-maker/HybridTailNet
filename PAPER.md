# HybridTailNet: Overcoming Information Bottlenecks and Missing Data via Dynamic Topological Metadata Tails

**Author:** Independent Researcher  
**Date:** November 2026  

## Abstract
Standard Multi-Layer Perceptrons (MLPs) propagate memoryless scalar activations across layers. When signals pass through narrow bottlenecks or encounter missing features (`NaN`), downstream neurons lose all structural context regarding the origin, path multiplicity, and completeness of the incoming signal. We introduce **HybridTailNet**, a dual-node neural architecture that augments scalar activations with a dynamically growing **topological metadata tail** tracking active pathway counts and network depth. Interspersed **Non-Neuron** units—powered by a single shared micro-network—read both layer statistics (`mean`, `max`, `min`) and the zero-padded tail to produce context-aware activations before collapsing the tail back to a unit state. Across synthetic structural logic benchmarks and a 10-seed evaluation on the *Spaceship Titanic* dataset, HybridTailNet achieves superior probability calibration (`LogLoss: 0.4348` vs. `0.4408`) while using **16.5% fewer parameters** (`1,778` vs. `2,129`) than a parameter-matched standard MLP.

## 1. Introduction
In classical artificial neural networks, every hidden unit computes a weighted linear combination followed by a non-linear activation. This design erases structural provenance: a zero activation caused by missing input data is indistinguishable from a zero activation resulting from the cancellation of multiple active signals. Furthermore, in architectures featuring narrow bottlenecks, compressing multi-dimensional activations into a single scalar creates severe information congestion.

To address this limitation without resorting to heavy graph networks or transformer attention, we propose **HybridTailNet**, which separates computation into two distinct node types:
1. **Standard Neurons (Accumulators):** Compute primary feature transformations while summing incoming path counters and appending depth markers.
2. **Non-Neurons (Context Compressors):** Periodically read the full accumulated state vector, modulate the primary signal using structural context, and reset (collapse) the tail.

## 2. Methodology
### 2.1 Dynamic State Representation
Instead of transmitting a flat scalar $v$, each connection carries a variable-length state vector:
$$\mathbf{s} = [v, \; t_1, \; t_2, \; \dots, \; t_k]$$
At the input layer, valid observed features are initialized with a unit marker $[x_i, 1.0]$, whereas missing values (`NaN`) are initialized as $[0.0, 0.0]$, carrying zero structural weight.

### 2.2 Tail Growth and Collapse
* **Accumulation:** Across standard neuron layers, incoming tails are summed element-wise and shifted with a newly appended depth marker `1.0`.
* **Renormalization in Non-Neurons:** A single lightweight micro-network $f_{\theta}$ is shared across all Non-Neurons in the network. It concatenates the node activation $v$, layer-level summary statistics ($\mu, \max, \min$), and the log-scaled, zero-padded tail vector (padded to fixed length $L_{\max}=10$):
  $$v_{\text{out}} = f_{\theta}\big([v, \; \mu, \; \max, \; \min, \; \text{Pad}_{L_{\max}-4}(\log(1 + \mathbf{t}))]\big)$$
  In bottleneck configurations, the Non-Neuron subsequently collapses the tail back to $[v_{\text{out}}, 1.0]$.

## 3. Experimental Evaluation
We evaluated HybridTailNet against parameter-matched standard MLPs on two benchmarks:
1. **3-Bottleneck Structural Logic (Data Scarcity, 20 training samples):** Alternating wide layers (`dim=8`) and single-unit bottlenecks (`dim=1`). HybridTailNet reduced test MSE by **41%** (`0.3340` vs. `0.5657`) and improved logical accuracy from `58.5%` to `70.0%`.
2. **Tabular Real-World Benchmark (*Spaceship Titanic*, 8,693 samples):** Using a fixed random blueprint of standard neurons and Non-Neurons (`1,778` parameters vs. `2,129` for MLP), HybridTailNet outperformed the baseline MLP in log-likelihood calibration (`0.4348` vs. `0.4408`).

## 4. Conclusion
Explicitly propagating and compressing topological path metadata allows compact neural networks to retain structural awareness and handle missing tabular features natively. Future work will explore integrating dynamic tails into recurrent and graph message-passing layers.
