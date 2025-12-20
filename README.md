# state_moment

The experimental codes and programs for the **Near-optimal simultaneous estimation of quantum state moments** project.

## Overview

This is a Github repository for the academic research of "Near-Optimal Simultaneous Estimation of Quantum State Moments." This work introduces a framework for the resource-efficient simultaneous estimation of quantum state moments via qubit reuse.

## Key Findings

*   **Resource Efficiency:** For an $m$-qubit quantum state $\rho$, the core circuit requires only $2m+1$ physical qubits and $\mathcal{O}(k)$ CSWAP gates to estimate moments up to the $k$-th order.
*   **Simultaneous Estimation:** The method achieves simultaneous estimation of the full hierarchy of moments $\text{Tr}(\rho^2), \dots, \text{Tr}(\rho^k)$, as well as arbitrary polynomial functionals and their observable-weighted counterparts.
*   **Near-Optimal Complexity:** The protocol achieves a near-optimal sample complexity of $\mathcal{O}(k \log k / \varepsilon^2)$.
*   **Applications:**
    *   The estimated moments yield tight bounds on a state's maximum eigenvalue.
    *   The protocol is applied in quantum virtual cooling to access low-energy states of the Heisenberg model.
*   **Experimental Validation:** The viability of the protocol was demonstrated on near-term quantum hardware by experimentally measuring higher-order Rényi entropy on a superconducting quantum processor.

## Paper

**Title:** Near-Optimal Simultaneous Estimation of Quantum State Moments  
**Authors:** Xiao Shi, Jiyu Jiang, Xian Wu, Jingu Xie, Hongshun Yao, Xin Wang  
**arXiv version:** [https://arxiv.org/abs/2509.24842](https://arxiv.org/abs/2509.24842)
