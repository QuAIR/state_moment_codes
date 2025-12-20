import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# ① Randomly generate a single-qubit Hermitian O
A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
O_single = A + A.conj().T

O_single = np.eye(2)
# Generate random parameters alpha, beta, gamma (Note: parameters must satisfy 0 <= alpha^2 + ... <= 1)
alpha, beta, gamma = np.random.rand(3)
print("alpha, beta, gamma:", alpha, beta, gamma)

# Construct a 7-qubit circuit (indexed 0-6) using the default.mixed device
dev = qml.device("default.mixed", wires=7)

# Define a function to generate a random 2x2 mixed density matrix
def random_density_matrix():
    """Generate a random 2x2 mixed density matrix."""
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = A @ np.conjugate(A.T)
    return rho / np.trace(rho)

@qml.qnode(dev)
def circuit(rho):
    # Initialize qubits 3, 4, 5, 6 to the mixed state rho
    for wire in range(3, 7):
        qml.QubitDensityMatrix(rho, wires=wire)
    
    # --- Initial State Preparation ---
    # Qubits 0 and 1 are initialized to |0> by default
    # Qubit 2 is constructed via an RY gate: alpha|0> + sqrt(1-alpha^2)|1>
    theta = np.arccos(alpha)  # cos(theta)=alpha, sin(theta)=sqrt(1-alpha^2)
    qml.RY(2 * theta, wires=2)
    
    # --- Gate Operations ---
    # 1. Apply Hadamard gate on qubit 0
    qml.Hadamard(wires=0)
    
    # 2. Use control pattern "101" (control qubits [0,1,2] expected state 1,0,1)
    # The default control requires all control bits to be 1, so we use control_values
    
    cswap_101 = qml.ctrl(qml.SWAP, control=[0, 1, 2], control_values=[1, 0, 1])
    cswap_101(wires=[3, 4])  # Perform SWAP on qubits 3 and 4

    # 3. Apply U1 gate on qubits [1, 2]
    # U1 4x4 matrix definition in subspace { |00>,|01>,|10>,|11> }:
    #   U1|00> = |00>
    #   U1|01> = beta         |01> - sqrt(1-beta^2)|10>
    #   U1|10> = sqrt(1-beta^2)|01> + beta         |10>
    #   U1|11> = |11>
    U1 = np.array([
        [1,           0,               0, 0],
        [0,         beta,   -np.sqrt(1 - beta**2), 0],
        [0, np.sqrt(1 - beta**2),         beta,      0],
        [0,           0,               0, 1]
    ])
    qml.QubitUnitary(U1, wires=[1, 2])
    

    # 4. Use control pattern "110" (control qubits [0,1,2] expected state 1,1,0)
    
    cswap_110 = qml.ctrl(qml.SWAP, control=[0, 1, 2], control_values=[1, 1, 0])
    cswap_110(wires=[3, 5])  # Perform SWAP on qubits 3 and 5


    # 5. Apply U2 gate on qubits [1, 2]
    # U2 matrix definition in subspace { |00>,|01>,|10>,|11> }:
    #   U2|00> = |00>, U2|01> = |01>
    #   U2|10> = gamma         |10> - sqrt(1-gamma^2)|11>
    #   U2|11> = sqrt(1-gamma^2)|10> + gamma         |11>
    U2 = np.array([
        [1,           0,              0, 0],
        [0,           1,              0, 0],
        [0,           0,         gamma, -np.sqrt(1 - gamma**2)],
        [0,           0, np.sqrt(1 - gamma**2),       gamma]
    ])
    controlled_Z = qml.ctrl(qml.PauliZ, control=[1, 2], control_values=[0, 1])
    controlled_Z(wires=0)
    qml.QubitUnitary(U2, wires=[1, 2])

    # 6. Use control pattern "111" (control qubits [0,1,2] all 1) to SWAP qubits [3, 6]
    cswap_111 = qml.ctrl(qml.SWAP, control=[0, 1, 2])
    cswap_111(wires=[3, 6])
    
    controlled_Z = qml.ctrl(qml.PauliZ, control=[1, 2], control_values=[1, 0])
    controlled_Z(wires=0)
    
    # --- Measurement ---
    # Finally measure PauliX on qubit 0, identity operation implied on others
    return qml.expval(
        qml.PauliX(wires=0)
    )


# Generate random mixed state density matrix (dimension 2x2)
rho = random_density_matrix()

# Run the circuit and output measurement results
result = circuit(rho)

# Draw circuit using qml.draw_mpl and display via matplotlib
# qml.draw_mpl(circuit)(rho)
print(qml.draw(circuit)(rho))

# Below is an example formula for the calculated result (based on your expression):

a1 = alpha**2
a2 = (1 - alpha**2) * beta**2 * np.trace(np.linalg.matrix_power(rho, 2))
a3 = (1 - alpha**2) * (1 - beta**2) * gamma**2 * np.trace(np.linalg.matrix_power(rho, 3))
a4 = (1 - alpha**2) * (1 - beta**2) * (1 - gamma**2) * np.trace(np.linalg.matrix_power(rho, 4))

calc_result = (
    alpha**2 -
    (1 - alpha**2) * beta**2 * np.trace(np.linalg.matrix_power(rho, 2) @ O_single) -
    (1 - alpha**2) * (1 - beta**2) * gamma**2 * np.trace(np.linalg.matrix_power(rho, 3) @ O_single) +
    (1 - alpha**2) * (1 - beta**2) * (1 - gamma**2) * np.trace(np.linalg.matrix_power(rho, 4) @ O_single)
)

print("Circuit output:", result)
print("Calculated result:", calc_result, a1, a2, a3, a4)
