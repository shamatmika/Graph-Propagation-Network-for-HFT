# LiqFlow: Physics-Guided Graph Propagation for HFT Order Book Dynamics
### (Still a work in progress, it is just a prototype for now)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?logo=pytorch)](https://pytorch.org)

Physics-guided **Graph Propagation Network (GPN)** for modelling liquidity dynamics in **high-frequency limit order books**.

LiqFlow represents the order book as a **4-D spatio-temporal graph** and learns how liquidity shocks propagate across price levels using **graph attention + physics-informed diffusion constraints**.

Built with **PyTorch** and **PyTorch Geometric** for GPU-accelerated training.

---

# Overview

Limit Order Books exhibit **structured spatial and temporal dynamics**:
large trades, cancellations, and liquidity shocks propagate across price levels.

Most deep learning models treat the LOB as a **time series**, ignoring this structure.

**LiqFlow addresses this by:**

* Representing the LOB as a **spatio-temporal graph**
* Using **Graph Attention Networks (GAT)** for liquidity propagation
* Enforcing **diffusion-advection PDE constraints** through a physics-informed loss

The result is a model that is not only predictive but **physically consistent with liquidity flow dynamics**.

---

# Model Architecture

## Pipeline

```
LOB Snapshot
      ↓
4D Graph Construction
(price level × time)
      ↓
Graph Attention Layers
(liquidity propagation)
      ↓
Physics Constraint Layer
(PDE diffusion loss)
      ↓
Prediction Head
(future liquidity / price movement)
```

---

# Core Concept

## LOB as a 4-D Graph

```
Nodes:
(price-level L, time-slice t)

Node features:
[volume, order imbalance, order flow]

Edges:
Spatial edges  → adjacent price levels
Temporal edges → same level across time
```

Graph attention propagates liquidity information across this structure.

---

## Physics Constraint

Liquidity dynamics are regularised using a **diffusion-advection PDE**:

```
∂_t φ = ∇·(D(σ)∇φ − v·φ + f(events))
```

Where:

* **φ** = liquidity field
* **D(σ)** = volatility-dependent diffusion
* **v** = drift (order flow direction)
* **f(events)** = external order events

This PDE models **how liquidity shocks propagate through the book**.

---

# Hybrid Training Objective

LiqFlow uses a **hybrid loss** combining prediction accuracy and physical consistency.

```
L = MSE(prediction, target)
  + λ · ||PDE residual||²
  + conservation penalties
```

Where the PDE residual is:

```
|| ∂_t φ − ∇·(D∇φ) ||²
```

This ensures that predicted liquidity states evolve according to **diffusion-like dynamics**.

---

# λ Sensitivity

The physics weight λ was swept over:

```
[0.001 ... 5.0]
```

Optimal value:

```
λ* ≈ 0.1
```

This value balances:

* prediction accuracy
* physics constraint enforcement

Large λ over-constrains the network, while small λ weakens physical consistency.

---

# Comparison With Existing Methods

| Method                 | Representation     | Liquidity Flow Modeling      | Structural Awareness |
| ---------------------- | ------------------ | ---------------------------- | -------------------- |
| CNN / LSTM (DeepLOB)   | Sequential         | Implicit                     | Low                  |
| Transformer LOB models | Temporal attention | Implicit                     | Medium               |
| Graph Neural Networks  | Graph topology     | Learned                      | High                 |
| **LiqFlow (proposed)** | 4-D graph          | **Physics-guided diffusion** | **Very High**        |

---

# Research Motivation

Financial markets exhibit **liquidity diffusion and shock propagation** similar to physical systems.

LiqFlow explores whether **physics-informed graph neural networks** can better capture these dynamics compared to purely data-driven approaches.



# Citation

If you use this repository, please cite:

```
@article{liqflow2026,
  title={LiqFlow: Physics-Guided Graph Propagation for High-Frequency Order Books},
  year={2026}
}
```

---

# License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software under the terms of the license. See the `LICENSE` file for full details.

Copyright © 2026 Shamatmika Raja

---

# Contributing

Pull requests and issues are welcome.
