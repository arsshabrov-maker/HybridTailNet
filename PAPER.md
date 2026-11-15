# GameTheoryNet: Noise-Resilient Multi-Agent Neural Layers Driven by Iterated Prisoner's Dilemma Strategies

**Author:** Independent Researcher  
**Date:** November 2026  

## Abstract
Conventional neural networks assume unconditional cooperation among hidden units, making them vulnerable to severe input corruption and adversarial sensor outliers. We introduce **GameTheoryNet**, a multi-agent neural architecture that replaces standard feed-forward layers with $R$ rounds of an **Iterated Prisoner's Dilemma** played by autonomous sub-networks ("players"). At each layer, players follow classic behavioral strategies (*Tit-for-Tat, Always-Truth, Always-Lie, Grudger, Pavlov*) to decide whether to transmit their true activation (**Cooperate**) or mask their output to zero (**Defect**), dynamically updating a pairwise **trust matrix** $R_{i,j}$. On benchmarks corrupted with **30% extreme sensor outliers**, a 5-player GameTheoryNet achieves **86.0% accuracy** compared to **81.5%** for a standard MLP while requiring **21% fewer parameters** (`1,038` vs. `1,313`).

## 1. Introduction
In standard deep learning architectures, every neuron blindly trusts and aggregates all incoming signals from the preceding layer. When input streams suffer from heavy-tailed noise or corrupted sensors, erroneous activations propagate unchecked to the output layer.

Drawing inspiration from Robert Axelrod's evolutionary tournaments on the Iterated Prisoner's Dilemma, we formulate forward propagation as a **multi-round reputation game**. By allowing specialized sub-networks to strategically withhold signals and track partner reliability across layers, the network learns to isolate corrupted pathways without explicit outlier-removal preprocessing.

## 2. Architecture & Mechanics
### 2.1 Strategic Actions: Truth vs. Lie
GameTheoryNet consists of $N=5$ player mini-networks interacting across $R=5$ sequential layers (rounds). At round $r$, player $i$ computes a candidate activation $v_i^{(r)}$ from input $X$ and trust-weighted partner states, then executes a binary strategic action $a_i^{(r)} \in \{0, 1\}$:
* **Truth ($a_i^{(r)} = 1$):** Transmits $v_i^{(r)}$ to the next round.
* **Lie ($a_i^{(r)} = 0$):** Masks its output to zero, functioning as an adaptive, reputation-driven dropout gate.

### 2.2 Pairwise Trust Matrix and Final Judge
After each round, every player $i$ updates its pairwise trust score toward partner $j$:
$$T_{i,j}^{(r)} = T_{i,j}^{(r-1)} + \mathbb{I}(a_j^{(r)} = 1) - \mathbb{I}(a_j^{(r)} = 0)$$
Incoming signals in round $r+1$ are modulated by $\sigma(T_{i,j}^{(r)})$. After $R$ rounds, a **Final Judge** sub-network aggregates the terminal player activations alongside the flattened trust matrix $T^{(R)}$ to produce the final prediction.

## 3. Experimental Results
We benchmarked GameTheoryNet against a parameter-matched Standard MLP (`1,313` parameters) across three regimes:
* **Clean Nonlinear Logic (XOR):** Standard MLP achieved `76.5%` vs. `66.0%` for GameTheoryNet (5 Mixed).
* **Noisy Environment (30% Extreme Outliers):** **GameTheoryNet (5 Mixed)** reached **86.0% accuracy** (`1,038` parameters), outperforming the Standard MLP (`81.5%`) by **+4.5%**.
* **Population Scaling:** Increasing population size to 30 players (`21,783` parameters) degraded mean accuracy to `61.50%` due to quadratic trust-matrix overfitting, confirming that compact 5-agent ecosystems provide optimal regularization.

## 4. Conclusion
Embedding game-theoretic reputation dynamics directly into hidden layers creates a natural inductive bias against noisy and corrupted inputs. Future extensions include evolutionary weight inheritance across rounds and integration with topological metadata tails.
