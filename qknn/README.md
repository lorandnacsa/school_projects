# Quantum k-Nearest Neighbors (QKNN) with Grover Search

This project implements a **Quantum k-Nearest Neighbors (QKNN)** classifier using **Qiskit**, **Grover’s algorithm**, and the **SWAP test** to perform similarity-based classification on the **Iris dataset**.  
The implementation combines classical preprocessing with quantum simulation to explore quantum-enhanced nearest-neighbor search.

---

## Overview

The workflow integrates:
- Classical data preprocessing and normalization
- Quantum amplitude encoding of feature vectors
- SWAP test–based similarity estimation
- Grover-based amplitude amplification to find the *k* nearest neighbors
- Majority voting for final classification

Parallel execution is used to speed up predictions across test samples.

---

## Algorithm Summary

1. **Data Encoding**
   - Features are normalized to the range `[0, 1]`
   - Each feature vector is amplitude-encoded into a quantum state

2. **Training State Preparation**
   - Training samples are encoded into a single superposition state:
     ```
     |ψ⟩ = (1 / √N) Σ |xᵢ⟩ |i⟩
     ```

3. **Similarity Estimation**
   - A **SWAP test** compares the test vector with all training vectors in superposition
   - Measurement outcomes are converted into distance estimates

4. **Grover Search**
   - An oracle marks training indices with distances below a chosen threshold
   - Grover amplification increases the probability of measuring the closest neighbors

5. **Classification**
   - The *k* nearest neighbors are selected
   - The final label is determined by majority vote

---

## Requirements

Install the required dependencies:

    pip install qiskit qiskit-aer numpy scikit-learn

---

## Limitations

Uses quantum simulation, not real quantum hardware

Circuit depth and qubit count scale poorly with dataset size

Grover oracle threshold selection is heuristic

Not suitable for large-scale datasets

