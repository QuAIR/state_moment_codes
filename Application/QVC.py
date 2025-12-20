# Filename: run_qvc_shots_scan.py

# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import scipy
from qiskit.circuit.library import Initialize
from qiskit_aer import AerSimulator
import time
import csv
import os

np.random.seed(42)
# --- Helper functions (same as previous version, no modification needed) ---
def real_value(rho, H, degree=3):
    list1, list2 = [], []
    temp1 = rho
    temp2 = H @ rho
    list2.append(float(np.trace(temp2).real))
    for ii in range(degree - 1):
        temp1 = temp1 @ rho
        temp2 = temp2 @ rho
        list1.append(float(np.trace(temp1).real))
        list2.append(float(np.trace(temp2).real))
    return list1, list2

def prepare_rho(qc, idx, state):
    init_gate = Initialize(state)
    qc.append(init_gate, idx)

def calculate_metrics(data_dict, op_list, term='Z'):
    if not data_dict:
        raise ValueError("Input dictionary cannot be empty")
    first_key = next(iter(data_dict.keys()))
    parts = first_key.split()
    l_val = len(parts[1])
    total_count = sum(data_dict.values())
    samples = []
    for key, count in data_dict.items():
        front_str, back_str = key.split()
        front_vals = [-1 if bit == '1' else 1 for bit in front_str]
        back_vals = [-1 if bit == '1' else 1 for bit in back_str]
        samples.append((front_vals, back_vals, count))
    weighted_sums_list1 = [0.0] * l_val
    weighted_sums_list2 = [0.0] * (l_val + 1)
    for front_vals, back_vals, count in samples:
        cum_prod = 1
        back_cum_prods = []
        for i in range(l_val):
            cum_prod *= back_vals[i]
            back_cum_prods.append(cum_prod)
        for j in range(l_val):
            weighted_sums_list1[j] += back_cum_prods[j] * count
        op_val = 0.0
        for op_item in op_list:
            op_str, coeff = op_item
            is_relevant_term = all(char in (term, 'I') for char in op_str)
            if not is_relevant_term:
                continue
            indices = [i for i, char in enumerate(op_str) if char == term]
            if not indices:
                continue
            prod = 1
            for idx in indices:
                prod *= front_vals[idx]
            op_val += coeff * prod
        weighted_sums_list2[0] += op_val * count
        for j in range(l_val):
            weighted_sums_list2[j + 1] += (op_val * back_cum_prods[j]) * count
    list1 = [s / total_count for s in weighted_sums_list1]
    list2 = [s / total_count for s in weighted_sums_list2]
    return list1, list2

def cswap(qc, idx1, idx2):
    for ii in range(len(idx1)):
        qc.cswap(0, idx1[ii], idx2[ii])

def circuit1(n, state, degree=3, pauli_str=None):
    if pauli_str is None:
        pauli_str = 'I' * n
    c = ClassicalRegister(degree - 1)
    crho = ClassicalRegister(n)
    q = QuantumRegister(3 * n + 1)
    qc = QuantumCircuit(q, c, crho)
    rho1 = list(range(1, n + 1))
    aux = list(range(n + 1, 2 * n + 1))
    rho2 = list(range(2 * n + 1, 3 * n + 1))
    prepare_rho(qc, rho1 + aux, state)
    for ii in range(degree - 1):
        qc.h(0)
        qc.reset(aux + rho2)
        prepare_rho(qc, rho2 + aux, state)
        cswap(qc, rho1, rho2)
        qc.h(0)
        qc.measure(q[0], c[ii])
        qc.reset(0)
    for qubit_idx, pauli_char in enumerate(pauli_str):
        if pauli_char == 'X':
            qc.h(rho1[qubit_idx])
        elif pauli_char == 'Y':
            qc.sdg(rho1[qubit_idx])
            qc.h(rho1[qubit_idx])
    qc.measure(rho1, crho)
    return qc

def generate_heisenberg_hamiltonian(n, J, h):
    if n < 2:
        return []
    hamiltonian = []
    for i in range(n - 1):
        for pauli in ['X', 'Y', 'Z']:
            op_list = ['I'] * n
            op_list[i] = pauli
            op_list[i+1] = pauli
            hamiltonian.append(("".join(op_list), J))
    for i in range(n):
        op_list = ['I'] * n
        op_list[i] = 'Z'
        hamiltonian.append(("".join(op_list), h))
    return hamiltonian

def run_single_simulation(J, h, beta, n, d, shots, run_index=1, total_runs=1):
    print(f"--- Starting simulation: n={n}, shots={shots}, run={run_index}/{total_runs} ---")
    
    Hamiltonian_str = generate_heisenberg_hamiltonian(n, J, h)
    
    H = SparsePauliOp.from_list(Hamiltonian_str).to_matrix()
    rho = scipy.linalg.expm(-beta * H)
    rho /= np.trace(rho)
    eigvals, eigvecs = scipy.linalg.eigh(rho)
    state = np.zeros(2 ** (2 * n), dtype=complex)
    for k, lam in enumerate(eigvals):
        aux_basis = np.zeros(2 ** n, dtype=complex)
        aux_basis[k] = 1
        state += np.sqrt(lam) * np.kron(aux_basis, eigvecs[:, k])

    simulator = AerSimulator()
    pauli_exists = {term: any(term in op for op, _ in Hamiltonian_str) for term in ['X', 'Y', 'Z']}

    list2_total = [0.0] * d
    list1_total = [0.0] * (d - 1)
    divid = 0

    H_parts = {'X': [], 'Y': [], 'Z': []}
    for op_str, coeff in Hamiltonian_str:
        if 'X' in op_str: H_parts['X'].append((op_str, coeff))
        if 'Y' in op_str: H_parts['Y'].append((op_str, coeff))
        if 'Z' in op_str: H_parts['Z'].append((op_str, coeff))
        
    for term in ['X', 'Y', 'Z']:
        if not pauli_exists[term]:
            continue
        
        H_term_list = H_parts[term]
        qc_term = circuit1(n=n, state=state, degree=d, pauli_str=term * n)
        job_term = simulator.run(qc_term, shots=shots)
        counts_term = job_term.result().get_counts()
        list1_term, list2_term = calculate_metrics(counts_term, H_term_list, term)
        
        divid += 1
        for i in range(d):
            list2_total[i] += list2_term[i]
            if i != d - 1:
                list1_total[i] += list1_term[i]

    if divid == 0:
        raise RuntimeError("No measurable X/Y/Z terms in Hamiltonian.")
    
    list1_total = [i / divid for i in list1_total]

    qvc_results = [0.0] * d
    qvc_results[0] = list2_total[0]
    for i in range(d - 1):
        qvc_results[i + 1] = list2_total[i + 1] / list1_total[i] if abs(list1_total[i]) > 1e-9 else float('nan')

    real_list1, real_list2 = real_value(rho=rho, H=H, degree=d)
    real_qvc_results = [0.0] * d
    real_qvc_results[0] = real_list2[0]
    for i in range(d - 1):
        real_qvc_results[i + 1] = real_list2[i + 1] / real_list1[i]

    return {
        "run_index": run_index,
        "n": n, "beta": beta, "J": J, "h": h, "shots": shots, "d": d,
        "qvc_results_sim": qvc_results,
        "qvc_results_real": real_qvc_results
    }

def save_all_results_to_csv(filename, all_results):
    if not all_results:
        print("No results to save.")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['run_index', 'n', 'beta', 'J', 'h', 'shots', 'd', 'order_k', 'term', 'simulated_value', 'real_value'])
        for result in all_results:
            d_val = result['d']
            writer.writerow([
                result['run_index'],
                result['n'], result['beta'], result['J'], result['h'], result['shots'], d_val,
                1, 'Tr(H rho)', result['qvc_results_sim'][0], result['qvc_results_real'][0]
            ])
            for i in range(d_val - 1):
                k = i + 2
                term_str = f"Tr(H rho^{k}) / Tr(rho^{k})"
                writer.writerow([
                    result['run_index'],
                    result['n'], result['beta'], result['J'], result['h'], result['shots'], d_val,
                    k, term_str, result['qvc_results_sim'][i+1], result['qvc_results_real'][i+1]
                ])
    print(f"\nAll results successfully saved to file: {filename}")


if __name__ == "__main__":
    total_start_time = time.time()
    
    # --- Simulation Parameters ---
    J = 1
    h = 1
    beta = 0.5
    d = 3
    NUM_RUNS = 3    # Number of repeated runs for each (n, shots) combination
    
    # Define the list of shots to test
    SHOTS_LIST = [10, 100, 1000, 10000, 100000, 1000000]
    # SHOTS_LIST = [10, 100]

    # --- Main Loop ---
    all_results = []
    for n_qubits in range(3, 7): # n runs from 3 to 6
        for shots_val in SHOTS_LIST: # Iterate through the shots list
            print(f"\n{'='*20} Processing n={n_qubits}, shots={shots_val} for {NUM_RUNS} runs {'='*20}\n")
            for i in range(1, NUM_RUNS + 1):
                result_data = run_single_simulation(
                    J=J, h=h, beta=beta, n=n_qubits, d=d, shots=shots_val,
                    run_index=i, total_runs=NUM_RUNS
                )
                all_results.append(result_data)

    # --- Save all results to CSV ---
    output_filename = f"qvc_error_vs_shots_scan_n3_to_n6_beta{beta}_3(j=h=1)_(run=3).csv"
    save_all_results_to_csv(output_filename, all_results)

    total_duration = time.time() - total_start_time
    print(f"\nAll simulations completed, total time elapsed: {total_duration:.2f} seconds")
