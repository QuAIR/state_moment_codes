# Qubit-Efficient Simultaneous Estimation of Nonlinear Quantum Properties

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2509.24842-b31b1b.svg)](https://arxiv.org/abs/2509.24842) This repository contains the official implementation and numerical simulation code for the paper **"Qubit-Efficient Simultaneous Estimation of Nonlinear Quantum Properties"** by Xiao Shi, Jiyu Jiang, Xian Wu, Jingu Xie, Hongshun Yao, and Xin Wang. 

## 📖 Overview

Estimating nonlinear properties of quantum states (such as moments $\mathrm{Tr}(O\rho^k)$ and Rényi entropies) is a central task in quantum information and many-body physics. However, traditional methods require either massive spatial overhead ($\mathcal{O}(kn)$ qubits) or exponentially scaling sample complexity. 

In this work, we propose a **unified, hardware-efficient circuit architecture** capable of extracting the entire sequence of nonlinear properties simultaneously. 

### Key Features
- **Drastic Resource Reduction**: Reduces the qubit requirement from $\mathcal{O}(kn)$ to $\mathcal{O}(n)$ using sequential state injection and mid-circuit measurements/resets.
- **Near-Optimal Sample Complexity**: Achieves a sample complexity of $\mathcal{O}(k \log k  C_O^2 / \epsilon^2)$, offering a rigorous quadratic improvement in the maximum degree $k$ compared to prior sequential methods.
- **Broad Practical Utility**: Supports simultaneous estimation of multiple polynomial functionals and bivariate state overlaps $\mathrm{Tr}[O(\rho\sigma)^j]$.

## ⚙️ Repository Structure

The codebase is organized simply and effectively, with all core simulation scripts contained within the `Application` directory:

```text
.
├── Application/
│   ├── QVC.py                  # Quantum Virtual Cooling (QVC) simulations
│   ├── bivariate_verify.py     # Verification of bivariate state overlaps 
│   ├── max_eigenvalue.py       # Estimation of maximum eigenvalue bounds
│   └── verify.py               # Core verification of the simultaneous estimation protocol
└── README.md

🚀 Installation
We recommend using a virtual environment (e.g., Conda) to run the simulations.

git clone [https://github.com/QUAIR/state_moment_codes.git](https://github.com/QUAIR/state_moment_codes.git)
cd state_moment_codes
conda create -n quantum_moments python=3.10
conda activate quantum_moments
# Install required packages (e.g., pennylane, numpy, scipy, matplotlib, quairkit)
pip install pennylane numpy scipy matplotlib quairkit

