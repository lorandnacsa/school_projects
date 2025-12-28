from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.visualization import plot_histogram
import numpy as np
import seaborn as sns
from statistics import mode
from sklearn import datasets, preprocessing 
from sklearn.model_selection import train_test_split
from qiskit.quantum_info import Statevector

def generate_oracle(n, threshold, result_arr):
    oracle = QuantumCircuit(n + 1, name="Oracle")
    for i in range(len(result_arr)):
        if result_arr[i][2] < threshold:
            binary_index = format(i, f"0{n}b")
            for bit, qubit in zip(binary_index, range(n)):
                if bit == '0':
                    oracle.x(qubit)
            oracle.mcx(list(range(n)), n)
            for bit, qubit in zip(binary_index, range(n)):
                if bit == '0':
                    oracle.x(qubit)
    return oracle

def amplitude_amplification(n, oracle):
    qc = QuantumCircuit(n, name="Grover")
    grover_operator = GroverOperator(oracle)
    qc.append(grover_operator, range(n))
    return qc

def find_k_minima(n, k, result_arr):
    index_register = QuantumRegister(n, 'index')
    target_qubit = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(n, 'result')
    qc = QuantumCircuit(index_register, target_qubit, classical_register)
    qc.h(index_register)
    threshold = np.percentile(result_arr[:, 2], 90)
    oracle = generate_oracle(n, threshold, result_arr)
    grover_circuit = amplitude_amplification(n + 1, oracle)
    qc.compose(grover_circuit, inplace=True)
    qc.measure(index_register, classical_register)
    backend = AerSimulator()
    job = backend.run(transpile(qc, backend), shots=10000)
    counts = job.result().get_counts()
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    k_min_indices = [int(state, 2) for state, _ in sorted_counts[:k]]
    return k_min_indices

def swap_test(N):
    a = QuantumRegister(N, 'a')
    b = QuantumRegister(N, 'b')
    d = QuantumRegister(1, 'd')
    qc_swap = QuantumCircuit(name = ' SWAP \nTest')
    qc_swap.add_register(a)
    qc_swap.add_register(b)
    qc_swap.add_register(d)
    qc_swap.h(d)
    for i in range(N):
        qc_swap.cswap(d, a[i], b[i])
    qc_swap.h(d)    
    return qc_swap

iris = datasets.load_iris()
feature_labels = iris.feature_names
class_labels = iris.target_names
X = iris.data
y = np.array([iris.target])
M = 4
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
min_max_scaler.fit(X)
X_normalized = min_max_scaler.transform(X)
N_train = 100
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y[0], train_size=N_train)
test_index = int(np.random.rand()*len(X_test)) 
phi_test = X_test[test_index]
for i in range(M):
    phi_i = [np.sqrt(phi_test[i]), np.sqrt(1- phi_test[i])]
    if i == 0:
        phi = phi_i
    else:
        phi = np.kron(phi_i, phi)    
print('Valid statevector: ', Statevector(phi).is_valid())
N = int(np.ceil(np.log2(N_train)))
psi = np.zeros(2**(M + N))
for i in range(N_train):
    i_vec = np.zeros(2**N)
    i_vec[i] = 1
    x = X_train[i, :]
    for j in range(M):
        dummy = [np.sqrt(x[j]), np.sqrt(1- x[j])]
        if j == 0:
            x_vec = dummy
        else:
            x_vec = np.kron(dummy, x_vec)
    psi_i = np.kron(x_vec, i_vec)
    psi += psi_i
psi /= np.sqrt(N_train)
assert Statevector(psi).is_valid(), "The statevector is not square-normalized."
index_reg = QuantumRegister(N, 'i')
train_reg = QuantumRegister(M, 'train')
test_reg = QuantumRegister(M, 'test')
p = QuantumRegister(1, 'similarity')
qknn = QuantumCircuit()
qknn.add_register(index_reg)
qknn.add_register(train_reg)
qknn.add_register(test_reg)
qknn.add_register(p)
qknn.initialize(psi, index_reg[0:N] + train_reg[0:M])
qknn.initialize(phi, test_reg)
qknn.append(swap_test(M), train_reg[0:M] + test_reg[0:M] + [p[0]])
meas_reg_len = N + 1
meas_reg = ClassicalRegister(meas_reg_len, 'meas')
qknn.add_register(meas_reg)
qknn.measure(index_reg[0::] + p[0::], meas_reg)
backend = AerSimulator()
job = backend.run(transpile(qknn, backend), shots=1000)
counts_knn = job.result().get_counts()
result_arr = np.zeros((N_train, 3))
for count in counts_knn:
    i_dec = int(count[1::], 2)
    phase = int(count[0], 2)
    if phase == 0:
        result_arr[i_dec, 0] += counts_knn[count]
    else:
        result_arr[i_dec, 1] += counts_knn[count]
for i in range(N_train):        
    prob_1 = result_arr[i][1]/(result_arr[i][0] + result_arr[i][1])
    result_arr[i][2] = 1 - 2*prob_1
n_qubits = int(np.ceil(np.log2(N_train)))
k = 10
k_min_dist_arr = find_k_minima(n_qubits, k, result_arr)
k_min_dist_arr = [idx for idx in k_min_dist_arr if idx < N_train]
if len(k_min_dist_arr) == 0:
    raise ValueError("Grover's search returned no valid indices. Adjust the threshold or check result array.")
y_pred = mode(y_train[k_min_dist_arr])
y_exp = y_test[test_index]
print('Predicted class of the test sample is {}.'.format(y_pred))
print('Expected class of the test sample is {}.'.format(y_exp))

def q_knn_module(test_index, psi, k=10):
    phi_test = X_test[test_index]
    for i in range(M):
        phi_i = [np.sqrt(phi_test[i]), np.sqrt(1- phi_test[i])]
        if i == 0:
            phi = phi_i
        else:
            phi = np.kron(phi, phi_i)    
    qknn = QuantumCircuit()
    qknn.add_register(index_reg)
    qknn.add_register(train_reg)
    qknn.add_register(test_reg)
    qknn.add_register(p)
    qknn.initialize(psi, index_reg[0:N] + train_reg[0:M])
    qknn.initialize(phi, test_reg)
    qknn.append(swap_test(M), train_reg[0:M] + test_reg[0:M] + [p[0]])
    meas_reg_len = N+1
    meas_reg = ClassicalRegister(meas_reg_len, 'meas')
    qknn.add_register(meas_reg)
    qknn.measure(index_reg[0::] + p[0::], meas_reg)
    backend = AerSimulator()
    job = backend.run(transpile(qknn, backend), shots=10000)
    counts_knn = job.result().get_counts()
    result_arr = np.zeros((N_train, 3))
    for count in counts_knn:
        i_dec = int(count[1::], 2)
        phase = int(count[0], 2)
        if phase == 0:
            result_arr[i_dec, 0] += counts_knn[count]
        else:
            result_arr[i_dec, 1] += counts_knn[count]
    for i in range(N_train):        
        prob_1 = result_arr[i][1]/(result_arr[i][0] + result_arr[i][1])
        result_arr[i][2] = 1 - 2*prob_1
    n_qubits = int(np.ceil(np.log2(N_train)))
    k = 10
    k_min_dist_arr = find_k_minima(n_qubits, k, result_arr)
    k_min_dist_arr = [idx for idx in k_min_dist_arr if idx < N_train]
    if len(k_min_dist_arr) == 0:
        raise ValueError("Grover's search returned no valid indices. Adjust the threshold or check result array.")
    y_pred = mode(y_train[k_min_dist_arr])
    y_exp = y_test[test_index]
    return y_pred, y_exp

y_pred_arr = []
y_exp_arr = [] 
for test_indx in range(len(X_test)):
    y_pred, y_exp = q_knn_module(test_indx, psi)
    y_pred_arr.append(y_pred)
    y_exp_arr.append(y_exp)
t = 0
f = 0
for i in range(len(X_test)):
    if y_pred_arr[i] == y_exp_arr[i]:
        t += 1
    else:
        f += 1
print('Model accuracy is {}%.'.format(t/(t+f) *100))
