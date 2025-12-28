from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from collections import Counter
import numpy as np

# Step 1: Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_labels = iris.feature_names
class_labels = iris.target_names

# Normalize the dataset using MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = min_max_scaler.fit_transform(X)

# Split the dataset into training and testing sets
N_train = 100  # Training dataset size
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, train_size=N_train)

# Step 2: Prepare the quantum state using rotations
def prepare_state(state_vector):
    """Prepare a quantum state using rotations."""
    
    num_qubits = len(state_vector)
    
    # Create quantum and classical registers
    qreg = QuantumRegister(num_qubits, name='q')
    creg = ClassicalRegister(num_qubits, name='c')
    qc = QuantumCircuit(qreg, creg)
    
    # Use RX and RY rotations to encode the state (Amplitude Encoding)
    for i, val in enumerate(state_vector):
        # Ensure the value passed is numeric (float)
        # Map value to angle using arccos for amplitude encoding
        angle = 2 * np.arccos(val)  # This ensures the value is used as an angle for rotation
        if isinstance(angle, (int, float)):  # Ensure angle is a real number
            qc.ry(angle, qreg[i])  # Apply rotation on the ith qubit
        else:
            print(f"Invalid angle for qubit {i}: {angle}")
    
    return qc, state_vector


# Step 3: Fidelity Calculation using Swap Test
def swap_test(test_state, train_state, N):
    """
    Performs a swap test to measure the fidelity between test_state and train_state.
    Returns the probability of measuring the ancilla in state |0>.
    
    `N`: Number of qubits in the registers.
    """
    # Quantum Registers
    a = QuantumRegister(N, 'a')  # Test state register
    b = QuantumRegister(N, 'b')  # Training state register
    d = QuantumRegister(1, 'd')  # Ancilla qubit
    
    # Classical Register to store the measurement result
    c = ClassicalRegister(1, 'c')
    
    # Create Quantum Circuit
    qc_swap = QuantumCircuit(a, b, d, c, name='Swap Test')
    
    # Initialize test and train states (using rotations instead of initialize)
    for i in range(N):
        qc_swap.ry(float(test_state[i]), a[i])  # Ensure angle is a float
        qc_swap.ry(float(train_state[i]), b[i])  # Ensure angle is a float
    
    # Apply Hadamard gate to the ancilla qubit
    qc_swap.h(d)
    
    # Apply controlled-swap gates between the test and train qubits
    for i in range(N):
        qc_swap.cswap(d, a[i], b[i])
    
    # Apply Hadamard gate to the ancilla qubit again
    qc_swap.h(d)
    
    # Measure the ancilla qubit
    qc_swap.measure(d, c)
    
    # Simulate the circuit
    simulator = AerSimulator()
    result = execute(qc_swap, backend=simulator, shots=1024).result()
    counts = result.get_counts(qc_swap)
    
    # Fidelity corresponds to the probability of measuring |0> on the ancilla qubit
    prob_0 = counts.get('0', 0) / 1024
    fidelity = 2 * prob_0 - 1
    return fidelity

# Step 4: Define Grover's Search for Top k Nearest Neighbors (k=2)
def create_maxima_oracle(fidelities, k):
    """Creates an oracle that marks the top k fidelities."""
    sorted_indices = np.argsort(fidelities)[-k:]  # Get indices of top k fidelities
    oracle = QuantumCircuit(len(fidelities), name='oracle')
    for i in sorted_indices:
        oracle.z(i)  # Flip the phase of the marked states
    return oracle

def grover_diffusion_operator(qc, qreg):
    """Applies the Grover diffusion operator (inversion about the mean)."""
    n = len(qreg)
    qc.h(qreg)
    qc.x(qreg)
    qc.h(qreg[n-1])
    qc.mct(qreg[:-1], qreg[n-1])  # Multi-controlled Toffoli gate
    qc.h(qreg[n-1])
    qc.x(qreg)
    qc.h(qreg)

def grover_search(fidelities, k):
    """Perform Grover's search to find the top k nearest neighbors based on fidelity."""
    n = len(fidelities)
    qreg = QuantumRegister(n)  # Quantum register to store the indices
    creg = ClassicalRegister(n)  # Classical register for measurement
    qc = QuantumCircuit(qreg, creg)
    
    # Apply the oracle and diffusion operator iteratively
    oracle = create_maxima_oracle(fidelities, k)
    qc.append(oracle, qreg)  # Add oracle to circuit
    
    # Apply Grover's diffusion operator
    grover_diffusion_operator(qc, qreg)
    
    # Measure the results
    qc.measure(qreg, creg)
    
    return qc

# Step 5: Run Grover’s Search
def run_grover_search(fidelities, k=2):
    # Apply Grover's search to find the top k neighbors
    grover_circuit = grover_search(fidelities, k)
    
    # Simulate the quantum circuit
    backend = AerSimulator()
    grover_circuit = transpile(grover_circuit, backend)
    qobj = assemble(grover_circuit, shots=1024)
    job = backend.run(qobj)
    result = job.result()
    
    # Get the measurement results
    counts = result.get_counts()
    
    # Get the top k fidelities
    sorted_indices = np.argsort(fidelities)[-k:]
    
    return sorted_indices, counts

# Step 6: Full QkNN Workflow with Grover's Search
def qknn_with_grover(train_data, train_labels, test_data, k=2):
    # Step 1: Prepare the training and test quantum circuits
    train_circuits = [prepare_state(train_data[i])[0] for i in range(len(train_data))]
    test_circuit = prepare_state(test_data[0])[0]

    # Step 2: Calculate fidelities between test state and each train state using the Swap Test
    fidelities = []
    for train_circuit in train_circuits:
        fidelity = swap_test(train_circuit, test_circuit, N=2)
        fidelities.append(fidelity)

    # Step 3: Run Grover’s search to find the top k nearest neighbors based on fidelity
    top_k_indices, counts = run_grover_search(fidelities, k=k)

    # Step 4: Get the labels of the top k nearest neighbors
    top_k_labels = [train_labels[i] for i in top_k_indices]

    # Step 5: Perform majority voting to classify the test state
    vote_counts = Counter(top_k_labels)
    majority_vote_label = vote_counts.most_common(1)[0][0]
    
    return majority_vote_label, top_k_indices

# Example usage:
predicted_label, top_k_indices = qknn_with_grover(X_train, y_train, X_test, k=2)

print(f"Predicted label for the test sample: {predicted_label}")
print(f"Top k nearest neighbors: {top_k_indices}")
