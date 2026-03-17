import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from quairkit.database import *
from quairkit.qinfo import *
import quairkit
from quairkit import Circuit, Hamiltonian

n = 2  # for single qubit state
r = 4  # rank


def get_kraus_operators(rho):
    d = rho.shape[0]
    zero = torch.tensor([1] + [0] * (d - 1), dtype=rho.dtype).reshape(-1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = torch.clamp(eigenvalues, min=0)  # 处理数值误差
    # 构造Kraus算符列表
    kraus_ops = []
    # 主Kraus算符: E_i = sqrt(p_i)|ψ_i⟩⟨0|
    for i in range(eigenvalues.shape[0]):
        p = eigenvalues[i]
        if p < 1e-8:  # 忽略数值噪声
            continue
        psi = eigenvectors[:, i].view(-1, 1)  # 转换为列向量
        E = torch.sqrt(p) * psi @ zero.conj().T
        kraus_ops.append(E)
    # 补充算符: E_rest = I - |0⟩⟨0|
    I = torch.eye(d, dtype=rho.dtype, device=rho.device)
    proj_zero = zero @ zero.conj().T
    kraus_ops.append(I - proj_zero)

    return kraus_ops


def initialize_rho(qc, qubit, rho, reset=False):
    """改成了利用 Kraus Channel 初始化任意的 rho"""
    if not isinstance(qubit, list):
        qubit = [qubit]
    if reset:
        qc.reset_channel(prob=[1, 0], qubits_idx=qubit)
    operators = get_kraus_operators(rho)
    operators = torch.stack(operators, dim=0)
    qc.kraus_channel(operators, qubit)
    return rho


rho1 = haar_density_operator(n, rank=n)
rho2 = haar_density_operator(n, rank=n)

qc = Circuit(r + 3)
qc.h(list(range(r)))
initialize_rho(qc, r, rho1)
initialize_rho(qc, r + 1, rho1)
initialize_rho(qc, r + 2, rho2)
for ii in range(r-1):
    qc.cswap([ii, r, r + 1])
    qc.cswap([ii, r + 1, r + 2])
    initialize_rho(qc, r + 1, rho1, reset=True)
    initialize_rho(qc, r + 2, rho2, reset=True)

qc.cswap([r-1, r, r + 2])
# qc.plot(print_code='True')

output_state = qc()

outcome_list = []
for ii in range(r):
    h_str = f"X{r - 1}"
    for jj in range(r-2, r-2-ii, -1):
        h_str = f"X{jj}, " + h_str
    h = Hamiltonian([[1., h_str]])
    outcome_list.append(output_state.expec_val(h))

expected_list = []
temp = rho1 @ rho2
for ii in range(1, r+1):
    expected_list.append(torch.trace(torch.matrix_power(temp, ii)))

print("All expected moments in ascending order: ", expected_list)
print("All observed moments in ascending order: ", outcome_list)
