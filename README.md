# quantum-lotto-trainer

ersuchen, ein Quantenmodell zur Vorhersage von Lottozahlen zu trainieren. Um die Abhängigkeiten zwischen aufeinanderfolgenden Vorhersagen effektiv zu modellieren, können Sie den Algorithmus "Recurrent Quantum Circuit" (RQC) verwenden. RQC ist eine Erweiterung des Quantum Circuit Born Machine (QCBM) und ermöglicht es, die Ausgabe einer vorherigen Vorhersage als Eingabe für die nächste Vorhersage zu verwenden.stelle nun noch sicher Nach jeder Simulation analysieren Sie die Ergebnisse und bestimmen, wie gut die Vorhersagen mit den tatsächlichen Daten übereinstimmen. Dies kann durch eine Kostenfunktion bewertet werden, die misst, wie nahe die Vorhersagen an den tatsächlichen Werten liegen.Verwenden Sie die Ergebnisse Ihrer Kostenfunktion, um die Parameter mithilfe eines Optimierungsalgorithmus anzupassen. Dies sollte darauf ausgerichtet sein, die Vorhersagegenauigkeit zu maximieren.Wiederholen Sie Dabei werden die aktualisierten Parameter verwendet, um die Genauigkeit der Vorhersagen schrittweise zu verbessern  import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit.quantum_info import Operator, SparsePauliOp, PauliList
from qiskit.circuit.library import RealAmplitudes, QFT
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.algorithms import QSVC
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit import transpile, assemble
from qiskit.circuit import Parameter
from qiskit.visualization import *
import math
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import *
import logging
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.exceptions import QiskitError
import traceback
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
import pickle
from qiskit_algorithms.optimizers import CRS
from qiskit_algorithms.time_evolvers import TimeEvolutionProblem, PVQD
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='qiskit_log.txt', filemode='w')
from qiskit.circuit.library import QAOAAnsatz
# Correct the URL in the QiskitRuntimeService initialization
service = QiskitRuntimeService(channel="ibm_quantum", token="e01a68b37bdc46b81fa8d52a5bd9ea52463f69b151a7527d8b833f2eb6423cd1d55f54372f49ee907d6e68f07326dd88dd772060346ae4f71d00b9e4973f1129")
#backend = service.backend('ibm_osaka')
#backend = service.backend('simulator_statevector')
backend = service.backend('ibmq_qasm_simulator')
#backend = service.backend('simulator_stabilizer')
#backend = service.backend('simulator_extended_stabilizer')
#print(service.instances())
#print(service.backends())
# Get a specific backend.

# Define parameters if they are part of the RealAmplitudes or other circuits
theta = Parameter('theta')
import numpy as np
import math

import numpy as np
import math

import numpy as np

import numpy as np

import numpy as np

def construct_graph(numbers, num_qubits):
    try:
        print("Starting to construct graph...")
        num_elements = np.prod(numbers.shape)
        num_columns = numbers.shape[1]
        num_rows = math.ceil(num_elements / num_columns)
        padded_numbers = np.pad(numbers, (0, num_rows * num_columns - num_elements), mode='constant', constant_values=0)
        reshaped_numbers = padded_numbers.reshape(num_rows, num_columns)
        print(f"Reshaped numbers into shape {reshaped_numbers.shape}")
        
        print("Reshaped numbers:\n", reshaped_numbers)
        
        if reshaped_numbers.shape[0] == 1:
            print("Warning: Only a single sample is available. Correlation coefficients cannot be computed.")
            # Handle the case when there is only a single sample
            # You can either return a default graph or use alternative methods to construct the graph
            return np.zeros((num_columns, num_columns))
        
        correlations = np.corrcoef(reshaped_numbers.T)
        print("Correlations shape:", correlations.shape)
        print("Correlations:\n", correlations)
        
        graph = np.zeros((num_columns, num_columns))

        for i in range(num_columns):
            for j in range(num_columns):
                if i != j and not np.isnan(correlations[i, j]) and abs(correlations[i, j]) > 0.5:
                    graph[i, j] = graph[j, i] = correlations[i, j]

        return graph

    except Exception as e:
        print(f"Error constructing graph: {e}")
        return None



def qgnn(graph, num_qubits):
    qc = QuantumCircuit(num_qubits)

    for i in range(graph.shape[0]):  # Iterate over the rows of the graph
        for j in range(graph.shape[1]):  # Iterate over the columns of the graph
            if i != j and abs(graph[i, j]) > 0.5:
                qc.cx(i, j)  # Controlled-X gate as an example
                qc.rz(graph[i, j], j)  # Rotation Z gate with parameter from graph

    return qc


def initialize_weights(num_qubits):
    # Calculate the total number of weights required
    num_rx_weights = num_qubits  # One RX gate weight per qubit
    num_crz_weights = int(num_qubits * (num_qubits - 1) / 2)  # One CRZ gate weight per pair of qubits
    total_required_weights = num_rx_weights + num_crz_weights

    # Initialize the weights with random values
    weights = np.random.rand(total_required_weights)
    print("Number of weights initialize_weights:", len(weights))
    return weights.tolist()  # Convert numpy array to list


from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
import numpy as np

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes


def qbm(feature_maps, num_qubits, epoch):
    """
    Construct a quantum circuit with adaptive encoding and a RealAmplitudes ansatz.
    Assumes feature_maps is a 2D NumPy array where the number of rows corresponds to the number of qubits.
    """
    if not isinstance(feature_maps, np.ndarray) or feature_maps.ndim != 2:
        raise ValueError("Feature maps must be a 2D NumPy array.")

    qc = QuantumCircuit(num_qubits)

    # Apply adaptive encoding based on the feature maps
    for i in range(num_qubits):
        for j in range(feature_maps.shape[1]):
            value = feature_maps[i, j]
            qc.rx(value, i)  # Apply RX gate with the feature map value
            if i < num_qubits - 1:  # Connecting to the next qubit in line
                qc.cx(i, i + 1)
                qc.rz(feature_maps[i, (j + 1) % feature_maps.shape[1]],
                      i + 1)  # Use modulo for circular feature connection
                qc.cx(i, i + 1)

    # Create a new RealAmplitudes circuit with default parameters
    ra_circuit = RealAmplitudes(num_qubits, reps=3)

    # Create a new set of parameters for an epoch
    new_params = ParameterVector(f'θ_{epoch}', length=num_qubits * (3 + 1))

    # Replace the parameters in the RealAmplitudes circuit
    param_dict = {old: new for old, new in zip(ra_circuit.parameters, new_params)}
    ra_circuit = ra_circuit.assign_parameters(param_dict, inplace=False)

    # Append the RealAmplitudes circuit to qc
    qc.append(ra_circuit, range(num_qubits))

    return qc

def adaptive_encoding(feature, feature_map):
    num_qubits = feature_map.num_qubits
    if isinstance(feature, dict):
        feature = list(feature.values())
    feature = np.array(feature).flatten()
    if len(feature) < num_qubits:
        feature = np.pad(feature, (0, num_qubits - len(feature)), 'constant', constant_values=0)
    input_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        input_circuit.ry(feature[i], i)
    qc = feature_map.compose(input_circuit)
    return qc
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def quantum_feature_selection(lottery_numbers, num_features):
    num_qubits = 6
    lottery_numbers = np.array(lottery_numbers)
    if lottery_numbers.ndim == 1:
        lottery_numbers = lottery_numbers.reshape(-1, 1)

    num_samples, num_features_available = lottery_numbers.shape
    if num_features > num_features_available:
        print("Adjusting num_features to available size:", num_features_available)
        num_features = num_features_available

    class_labels = np.zeros(num_samples)
    selector = SelectKBest(f_classif, k=num_features)
    selector.fit(lottery_numbers, class_labels)
    selected_features = lottery_numbers[:, selector.get_support(indices=True)]

    # Debugging output
    print("Shape of selected_features:", selected_features.shape)
    return selected_features



from qiskit.circuit.library import ZGate
from qiskit_algorithms.optimizers import SPSA  # Using SPSA as an example optimizer
import numpy as np
def cost_function(actual, predicted):
    """
    Computes the cost between the actual and predicted values.

    Args:
        actual (np.ndarray): The actual values.
        predicted (np.ndarray): The predicted values.

    Returns:
        float: The cost value.
    """
    # Ensure that the actual and predicted arrays have the same shape
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")

    # Calculate the mean squared error (MSE)
    mse = np.mean((actual - predicted) ** 2)

    return mse

from qiskit.circuit import ParameterVector

def update_weights(qc, weights, num_qubits, learning_rate, lottery_data, predicted_probabilities):
    if np.isscalar(weights):
        weights = np.array([weights])
    num_weights = len(weights)
    lottery_numbers = np.array(lottery_data)
    # Rest of the code...

    # Create a parameterized quantum circuit
    params = ParameterVector('theta', num_weights)
    parameterized_qc = qc.assign_parameters(dict(zip(qc.parameters, params)))

    # Define the objective function for the optimizer
    def objective_function(params):
        # Evaluate the parameterized quantum circuit with the given parameters
        bound_params = {param: val for param, val in zip(params, params)}
        parameterized_qc.assign_parameters(bound_params)
        qc_state = parameterized_qc.bind_parameters(bound_params)
        
        # Example: Calculate cost function based on quantum state and expected probabilities
        predicted_state = np.abs(np.dot(qc_state, weights))
        cost_value = cost_function(lottery_numbers, predicted_state)
        return cost_value

    # Initialize the SPSA optimizer with advanced settings
    spsa = SPSA(maxiter=100, learning_rate=learning_rate, second_order=True)

    # Perform optimization to update the weights
    initial_point = np.array(weights)
    result = spsa.optimize(num_weights, objective_function, initial_point=initial_point)
    updated_weights = result.x

    return updated_weights
from qiskit.quantum_info import Operator


def update_feature_maps(qc, feature_maps, num_qubits, learning_rate):
    try:
        # Ensure the correct number of qubits
        print("Number of num_qubits in update_feature_maps:", num_qubits)
        num_qubits = 6  # You might want to remove this line if num_qubits is provided as an argument
        
        cost_operator = ZGate() ^ num_qubits

        # Define the objective function for the optimizer
        def objective_function(params):
            # Evaluate the parameterized quantum circuit with the given parameters
            parameterized_fm = qc.assign_parameters(dict(zip(qc.parameters, params)))
            expectation_value = parameterized_fm.assign_parameters({param: val for param, val in zip(params, params)}).compute_expectation_value(Operator(cost_operator))

            # Example: Calculate cost value based on expectation value
            cost_value = expectation_value

            return cost_value

        # Initialize the SPSA optimizer
        optimizer = SPSA(maxiter=10, learning_rate=learning_rate)

        # Perform optimization to update the feature maps
        result = optimizer.optimize(len(feature_maps), objective_function, initial_point=feature_maps)
        updated_feature_maps = result.x

        return updated_feature_maps
    except Exception as e:
        # Handle any errors that might occur during the update
        print(f"Error updating feature maps: {e}")
        # Optionally raise the exception if the function is critical and should not fail silently
        raise ValueError("Failed to update feature maps due to an error.") from e

"""
def initialize_feature_maps(num_qubits, reps=2, entanglement='linear'):
    feature_map = ZZFeatureMap(num_qubits, reps=reps, entanglement=entanglement)
    print("Feature map initialized with shape:", feature_map.num_qubits)
    return feature_map"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit.quantum_info import SparsePauliOp

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA, QNSPSA
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

def rescale_numbers(numbers, scale=49):
    # Rescale normalized numbers to the original range
    return {key: int(value * scale) for key, value in numbers.items()}


def safe_int_conversion(x):
    try:
        return int(x, 2) if isinstance(x, str) and set(x) <= {'0', '1'} else float(x)
    except (ValueError, TypeError):
        return 0  # Fallback to 0 if conversion fails

def normalize(probs):
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}

def format_lottery_numbers(normalized_values, max_value=49):
    normalized_values = [safe_int_conversion(val) for val in normalized_values]
    normalized_values = np.array(normalized_values, dtype=float)
    if normalized_values.size == 0:
        return {}
    lottery_numbers = np.round(normalized_values * max_value).astype(int)
    return {f'x{i + 1}': num for i, num in enumerate(lottery_numbers)}

def cost_function(actual_data, next_actual_data, predicted_probabilities, num_qubits, epoch, max_epoch):
    # Ensure num_qubits does not exceed the length of actual data
    num_qubits = min(num_qubits, len(actual_data), len(next_actual_data))
    
    normalized_probs = normalize(predicted_probabilities)
    predicted_values = [
        safe_int_conversion(state) for state, prob in sorted(normalized_probs.items(), key=lambda item: item[1], reverse=True)[:num_qubits]
        ]
    
        # Normalize predicted values to the range of actual data
    if predicted_values:
        max_predicted = max(predicted_values)
        min_predicted = min(predicted_values)
        normalized_predicted = np.array([(x - min_predicted) / (max_predicted - min_predicted) for x in predicted_values])
    else:
        normalized_predicted = np.array([])
    
    # Ensure actual data is correctly passed and processed
    if isinstance(actual_data, dict):
        actual_data = list(actual_data.values())
    if isinstance(next_actual_data, dict):
        next_actual_data = list(next_actual_data.values())
    
    actual_lottery_format = format_lottery_numbers(actual_data)
    next_actual_lottery_format = format_lottery_numbers(next_actual_data)
    Predicted_values_lottery_format = format_lottery_numbers(normalized_predicted)
    mse = np.mean([(predicted - actual) ** 2 for predicted, actual in zip(normalized_predicted, next_actual_data)]) if next_actual_data.size > 0 else 0
    entropy = -sum(prob * np.log2(prob) if prob > 0 else 0 for prob in normalized_probs.values())
    penalties = sum(prob for state, prob in normalized_probs.items() if not (1 <= safe_int_conversion(state) <= 49))
    penalty_factor = np.exp(-epoch / max_epoch)
    total_penalty = penalty_factor * penalties
    total_cost = mse + entropy + total_penalty
    
    print(f"Actual (Lottery Format): {actual_lottery_format}")
    print(f"Next Actual (Lottery Format): {next_actual_lottery_format}")
    print(f"Predicted Values(Lottery Format): {Predicted_values_lottery_format}")   
    print(f"Normalized Predicted Values: {normalized_predicted}")
    print(f"MSE: {mse}")
    print(f"Entropy: {entropy}")
    print(f"Penalties: {penalties}")
    
    return total_cost


def calculate_entropy(probabilities):
    """ Calculate entropy based on probabilities. """
    return -sum(prob * np.log2(prob) if prob > 0 else 0 for prob in probabilities.values())

def calculate_penalties(probabilities, max_value):
    """ Calculate penalties for predictions out of range, using normalized probabilities. """
    return sum(prob for state, prob in probabilities.items() if not (0.02040816 <= state / max_value <= 1))


def evaluate_mse(predictions, actuals):
    """Calculate Mean Squared Error to evaluate model performance."""
    return np.mean([(pred - act) ** 2 for pred, act in zip(predictions, actuals)])
def initialize_feature_maps(num_qubits, num_features):
    feature_map = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        feature_map.ry(np.pi/4, q)  # Basisrotationen für Features
        feature_map.rz(np.pi/8, q)  # Feinabstimmung der Phasen
    return feature_map

from qiskit_algorithms.optimizers import CRS
from qiskit_algorithms.time_evolvers import PVQD, TimeEvolutionProblem
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Estimator

def train_model(lottery_numbers, num_epochs, num_features, learning_rate):
    if isinstance(lottery_numbers, np.ndarray):
        # Convert numpy array to DataFrame with appropriate column names
        lottery_numbers = pd.DataFrame(lottery_numbers, columns=[f'num_{i}' for i in range(lottery_numbers.shape[1])])
        print("Converted lottery numbers to DataFrame:")
        print(lottery_numbers)

    num_qubits = num_features
    # Assuming initialize_feature_maps function exists and is defined elsewhere
    feature_maps = initialize_feature_maps(num_qubits, num_features)
    ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=9)
    # Normalize and convert to DataFrame within the normalize_lottery_numbers function
    normalized_lottery_numbers = lottery_numbers
    print("Normalized lottery numbers:")
    print(normalized_lottery_numbers)
    reps = 9
    optimizer = CRS()  # Using CRS global optimizer
    initial_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)

    losses = []
    predicted_numbers = []
    results_summary = []
    simulator = AerSimulator(method="statevector")

    for epoch in range(num_epochs):
        if epoch == 0:
            current_params = initial_params
        else:
            current_params = optimal_params

        previous_predictions = None

        for index in range(len(normalized_lottery_numbers) - 1):
            row = normalized_lottery_numbers.iloc[index]
            next_row = normalized_lottery_numbers.iloc[index + 1]

            if previous_predictions is not None:
                row = previous_predictions  # Ensure this is in the correct format

            row_values = row.values if isinstance(row, pd.Series) else row
            next_row_values = next_row.values if isinstance(next_row, pd.Series) else next_row

            qc = QuantumCircuit(num_qubits)
            encoded_circuit = adaptive_encoding(row_values, feature_maps)
            qc.compose(encoded_circuit, inplace=True)

            ra_circuit = RealAmplitudes(num_qubits, reps=reps)
                        # Hinzufügen von Controlled und Toffoli Gates
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            if num_qubits > 2:
                qc.ccx(0, 1, 2)

            qc.append(ra_circuit, range(num_qubits))
            qc.measure_all()
                        # Define the Hamiltonian for QAOA
            pauli_string = 'I' * num_qubits
            coefficients = np.array([1.0])
            pauli_list = PauliList([pauli_string])
            hamiltonian = SparsePauliOp(pauli_list, coeffs=coefficients)
            # Initialize the QAOA circuit


            # Initialize the optimizer
            optimizer = QNSPSA(SPSA(maxiter=num_epochs), maxiter=num_epochs)
            qaoa_circuit = QAOAAnsatz(hamiltonian, reps=9)
            qaoa_circuit.compose(qc, front=True, inplace=True)

            transpiled_circuit = transpile(qaoa_circuit, simulator)
            parameter_binds = {param: value for param, value in zip(qaoa_circuit.parameters, current_params)}
            qobj = assemble(transpiled_circuit, shots=1000, parameter_binds=[parameter_binds])
            result = simulator.run(qobj).result()
            counts = result.get_counts()
            total_counts = sum(counts.values())
            predicted_probabilities = {int(bs, 2): count / total_counts for bs, count in counts.items()}

            # Calculate loss here before updating parameters
            loss = cost_function(row_values, next_row_values, predicted_probabilities, num_qubits, epoch, num_epochs)

            # Now call update_parameters with the correctly calculated loss
            optimal_params = update_parameters(qc, current_params, loss, optimizer, simulator, learning_rate)

            # Update the circuit with optimal parameters
            optimized_circuit = transpile(qaoa_circuit.assign_parameters(parameter_binds), simulator)
            optimized_result = simulator.run(optimized_circuit, shots=1000).result()
            optimized_counts = optimized_result.get_counts()
            optimized_probabilities = {int(bs, 2): count / sum(optimized_counts.values()) for bs, count in optimized_counts.items()}
            predicted_numbers = sorted(optimized_probabilities, key=lambda x: optimized_probabilities[x], reverse=True)[:num_qubits]
            predicted_numbers_dict = {f'x{i + 1}': num + 1 for i, num in enumerate(predicted_numbers)}

            losses.append(loss)
            predicted_numbers.append(predicted_numbers_dict)
            previous_predictions = optimized_probabilities
            results_summary.append({
                'epoch': epoch,
                'index': index,
                'actual': row.values,
                'next_actual': next_row.values,
                'predicted': predicted_numbers_dict,
                'loss': loss
            })

            print(f"Epoch {epoch}, Index {index}: Actual - {row.values}, Next Actual - {next_row.values}, Predicted - {predicted_numbers_dict}")

    avg_loss_per_epoch = pd.DataFrame(results_summary).groupby('epoch')['loss'].mean()
    avg_loss_per_epoch.plot(kind='bar', color='skyblue')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

    return [], [], [], losses, predicted_numbers, optimal_params



def update_parameters(qc, params, loss, optimizer, simulator_backend, learning_rate=0.001):
    shift = np.pi / 2
    gradients = np.zeros_like(params)

    for i in range(len(params)):
        # Plus parameter shift
        params_plus = np.array(params, copy=True)
        params_plus[i] += shift
        qc_plus = qc.assign_parameters(dict(zip(qc.parameters, params_plus)))
        counts_plus = simulate_quantum_circuit(qc_plus, simulator_backend)

        # Minus parameter shift
        params_minus = np.array(params, copy=True)
        params_minus[i] -= shift
        qc_minus = qc.assign_parameters(dict(zip(qc.parameters, params_minus)))
        counts_minus = simulate_quantum_circuit(qc_minus, simulator_backend)

        # Total counts
        total_counts_plus = sum(counts_plus.values())
        total_counts_minus = sum(counts_minus.values())

        # Probability distributions
        probs_plus = {k: v / total_counts_plus for k, v in counts_plus.items()}
        probs_minus = {k: v / total_counts_minus for k, v in counts_minus.items()}

        # Find the key with max value in both distributions
        key_max_plus = max(probs_plus, key=probs_plus.get)
        key_max_minus = max(probs_minus, key=probs_minus.get)

        # Gradient approximation - considering the highest count state
        gradients[i] = probs_plus.get(key_max_plus, 0) - probs_minus.get(key_max_minus, 0)

        # Updating gradient approximation to consider sum difference of probability distributions
        states_in_both = set(probs_plus.keys()) & set(probs_minus.keys())
        gradients[i] += sum(abs(probs_plus[state] - probs_minus[state]) for state in states_in_both)

    updated_params = params - learning_rate * gradients * loss  # Adjusting by loss magnitude
    return updated_params


def simulate_quantum_circuit(qc, simulator_backend):
    transpiled_circuit = transpile(qc, simulator_backend)
    qobj = assemble(transpiled_circuit, shots=1000)
    result = simulator_backend.run(qobj).result()
    counts = result.get_counts()
    return counts


def evaluate_model(circuits, weights_list, feature_maps, test_data, sim):
    print("Evaluating model.")
    num_qubits = test_data.shape[1]
    predicted_numbers = []
    transpiled_circuits = []
    parameter_binds = []

    if isinstance(test_data, np.ndarray):
        # Convert the numpy array to a DataFrame
        test_data = pd.DataFrame(test_data)

    # Initialize the optimizer
    optimizer = CRS()

    for index, lottery_numbers in test_data.iterrows():
        normalized_lottery_numbers = normalize_lottery_numbers(lottery_numbers, max_value=49)
        print(f"Normalized lottery numbers for index {index}: {normalized_lottery_numbers}")

        qc_test = QuantumCircuit(num_qubits)
        if index < len(circuits):
            qc_test.compose(circuits[index], inplace=True)
        else:
            print(f"Skipping circuit for index {index} as it is out of range.")
            continue

        qbm_circuit = qbm(feature_maps, num_qubits, index)
        qc_test.compose(qbm_circuit, inplace=True)

        adaptive_encoding_circuit = adaptive_encoding(normalized_lottery_numbers, feature_maps)
        qc_test.compose(adaptive_encoding_circuit, inplace=True)

        ra_circuit = RealAmplitudes(num_qubits, reps=6)
        qc_test.append(ra_circuit, range(num_qubits))
        qc_test.measure_all()

        transpiled_circuit = transpile(qc_test, sim)
        transpiled_circuits.append(transpiled_circuit)
        print(f"Transpiled circuit parameters for index {index}: {transpiled_circuit.parameters}")
        print(f"Weights for index {index}: {weights_list[index]}")

        param_binds = {str(param): weight for param, weight in zip(transpiled_circuit.parameters, weights_list[index])}
        parameter_binds.append(param_binds)

    if not transpiled_circuits:
        print("No valid transpiled circuits found.")
        return []

    # Use Variational Time Evolution and VQE to optimize the parameters
    pauli_string = 'I' * num_qubits
    coefficients = np.array([1.0])
    pauli_list = PauliList([pauli_string])
    cost_operator = SparsePauliOp(pauli_list, coeffs=coefficients)

    ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=3)
    vqte = TimeEvolutionProblem(ansatz, optimizer, cost_operator)
    time_evolution = vqte.evolve(cost_operator, time=1.0, num_time_slices=1)

    optimal_params = time_evolution.optimal_point
    optimized_circuit = ansatz.assign_parameters(optimal_params)

    # Run VQE with the optimized circuit
    estimator = Estimator()
    vqe = VQE(ansatz=optimized_circuit, optimizer=optimizer, estimator=estimator)
    vqe_result = vqe.compute_minimum_eigenvalue(cost_operator)

    final_params = vqe_result.optimal_point
    final_circuit = optimized_circuit.assign_parameters(final_params)
    final_circuit.measure_all()
    final_circuit = transpile(final_circuit, sim)

    # Ensure each circuit is assigned parameters correctly and run them as a batch
    bound_circuits = [final_circuit.assign_parameters(bind) for bind in parameter_binds]
    job = sim.run(bound_circuits, shots=1000)
    results = job.result()

    for i, circuit in enumerate(bound_circuits):
        counts = results.get_counts(circuit)
        print(f"Counts for circuit {i}: {counts}")
        counts_dict = defaultdict(int)
        for k, v in counts.items():
            counts_dict[int(k, 2)] += v
        total_counts = sum(counts_dict.values())
        predicted_probabilities = {k: v / total_counts for k, v in counts_dict.items()}
        sorted_indices = np.argsort(list(predicted_probabilities.values()))[::-1]
        top_predictions = [list(predicted_probabilities.keys())[i] for i in sorted_indices[:6]]
        predicted_numbers.append(top_predictions)

    return predicted_numbers
import seaborn as sns
from sklearn.manifold import TSNE

# Visualisierung und Interpretation
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import logging
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.exceptions import QiskitError
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import logging
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.exceptions import QiskitError
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
def visualize_results(losses, predicted_numbers, test_data, feature_maps, weights, sim):
    lottery_numbers_test = preprocess_data(test_data)
    if lottery_numbers_test is None or lottery_numbers_test.size == 0:
        print("Error: No data available for visualization.")
        return

    num_qubits = feature_maps.shape[0]

    all_predicted_probabilities = []
    graph = None
    qc_test = None

    for index, data_point in enumerate(lottery_numbers_test):
        graph = construct_graph(data_point[np.newaxis, :], num_qubits)

        qc_test = QuantumCircuit(num_qubits, num_qubits)
        qc_test.compose(qgnn(graph, num_qubits), inplace=True)
        qc_test.compose(qbm(feature_maps, num_qubits, index), inplace=True)
        qc_test.compose(adaptive_encoding(data_point, feature_maps), inplace=True)

        qc_test.measure(range(num_qubits), range(num_qubits))
        transpiled_circuit = transpile(qc_test, sim)

        parameter_values = dict(zip(transpiled_circuit.parameters, weights[index]))
        bound_circuit = transpiled_circuit.assign_parameters(parameter_values)

        job = sim.run(bound_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        predicted_probabilities = np.array(
            [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(2 ** num_qubits)]) / 1000
        all_predicted_probabilities.append(predicted_probabilities)

    visualize_data(losses, all_predicted_probabilities, graph, qc_test)


def visualize_data(losses, all_predicted_probabilities, graph, qc_test):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(losses)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 2)
    mean_probabilities = np.mean(all_predicted_probabilities, axis=0)
    plt.bar(range(len(mean_probabilities)), mean_probabilities)
    plt.title('Predicted Probabilities')
    plt.xlabel('Lottery Number')
    plt.ylabel('Probability')

    plt.subplot(2, 3, 3)
    if graph is not None:
        sns.heatmap(graph, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')

    plt.subplot(2, 3, 6)
    if qc_test is not None:
        qc_test.draw('mpl')
        plt.title('Quantum Circuit')

    plt.tight_layout()
    plt.show()


def preprocess_data(lottery_data):
    if not lottery_data:
        print("Warning: No data provided to preprocess.")
        return None

    if isinstance(lottery_data, dict):
        lottery_data = np.array(list(lottery_data.values()))

    elif isinstance(lottery_data, list):
        lottery_data = np.array(lottery_data)

    elif isinstance(lottery_data, pd.DataFrame):
        lottery_data = lottery_data.values

    else:
        print(f"Warning: data type not supported: {type(lottery_data)}")
        return None

    print(f"Processed data: {lottery_data}, type: {type(lottery_data)}")
    return normalize_lottery_numbers(lottery_data, max_value=49)


from collections.abc import Iterable

"""
def normalize_lottery_numbers(lottery_data, max_value=49):
    if isinstance(lottery_data, dict):
        lottery_data = np.array(list(lottery_data.values()))
    elif isinstance(lottery_data, np.ndarray) and lottery_data.dtype == object:
        max_length = max(len(d) for d in lottery_data if isinstance(d, Iterable))
        padded_arrays = [np.pad(np.array(list(d.values())), (0, max_length - len(d)), mode='constant') for d in
                         lottery_data if isinstance(d, Iterable)]
        lottery_data = np.stack(padded_arrays)
    else:
        return lottery_data"""

def normalize_lottery_numbers(lottery_data, max_value=49):
    if isinstance(lottery_data, dict):
        lottery_data = np.array(list(lottery_data.values()))
    elif isinstance(lottery_data, np.ndarray) and lottery_data.dtype == object:
        max_length = max(len(d) for d in lottery_data)
        padded_arrays = [np.pad(np.array(list(d.values())), (0, max_length - len(d)), mode='constant') for d in lottery_data]
        lottery_data = np.stack(padded_arrays)

    # Ensure lottery_data is a numpy array before calling astype
    lottery_data = np.array(lottery_data)
    normalized_values = lottery_data.astype(float) / max_value
    return normalized_values

def save_model(circuits, weights_list, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({'circuits': circuits, 'weights': weights_list}, f)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def initialize_simulator(service):
    # List of possible simulators to try
    simulator_options = [#'ibm_sherbrooke', 

        'ibmq_qasm_simulator',  # Typically used ID for IBM QASM simulators
        'simulator_statevector',
        'simulator_extended_stabilizer'
    ]

    # Try each simulator in order until one succeeds
    for simulator_name in simulator_options:
        try:
            sim = service.backend(simulator_name)
            logging.info(f"Simulator {simulator_name} initialized successfully.")
            return sim
        except Exception as e:
            logging.error(f"Error accessing simulator {simulator_name}: {e}")

    # If all else fails, fall back to the default Aer simulator
    logging.warning("Falling back to default AerSimulator.")
    return AerSimulator()

if __name__ == '__main__':
    # Load data from a CSV file
    lottery_data = pd.read_csv("lotto-zaheln_14.csv", delimiter=';', header=0)
    print("Original lottery_data:\n", lottery_data)
    service = QiskitRuntimeService()
    sim = initialize_simulator(service)
    #sim = AerSimulator()
    #sim = AerSimulator(method='extended_stabilizer')
    # Normalize the lottery data and maintain its DataFrame structure
    normalized_lottery_data = normalize_lottery_numbers(lottery_data)
    print("Normalized lottery_data:\n", normalized_lottery_data)

    # Split the normalized data into training and testing sets, ensuring the structure is maintained
    train_data, test_data = train_test_split(normalized_lottery_data, test_size=0.2, random_state=42)

    # Print the shapes and some data to verify everything is correct
    print("Shape of normalized_lottery_data:", normalized_lottery_data.shape)
    print("Shape of train_data:", train_data.shape)
    print("Train_data sample:\n", train_data[:5])
    print("Shape of test_data:", test_data.shape)
    print("Test_data sample:\n", test_data[:5])

    num_epochs = 15
    num_features = 6
    learning_rate = 0.002
    num_qubits = 6
    optimizer = SPSA(maxiter=num_epochs)
    # Convert train_data DataFrame to a NumPy array
    train_data_np = train_data

    # Unpack the values returned by train_model
    circuits, weights_list, feature_maps, losses, predicted_numbers, parameter_binds = train_model(train_data_np, num_epochs, num_features, learning_rate)
    print("Predicted numbers:", predicted_numbers)
    save_model(circuits, weights_list)
    # Pass all necessary arguments to the evaluate_model function
    predicted_numbers_test = evaluate_model(circuits, weights_list, feature_maps, test_data, sim)

    # Print the predicted numbers
    print("Predicted numbers:", predicted_numbers_test)
    lottery_numbers_test = train_data_np = train_data




    visualize_results(losses, lottery_numbers_test, predicted_numbers, feature_maps, weights_list, sim)

## Collaborate with GPT Engineer

This is a [gptengineer.app](https://gptengineer.app)-synced repository 🌟🤖

Changes made via gptengineer.app will be committed to this repo.

If you clone this repo and push changes, you will have them reflected in the GPT Engineer UI.

## Tech stack

This project is built with React and Chakra UI.

- Vite
- React
- Chakra UI

## Setup

```sh
git clone https://github.com/GPT-Engineer-App/quantum-lotto-trainer.git
cd quantum-lotto-trainer
npm i
```

```sh
npm run dev
```

This will run a dev server with auto reloading and an instant preview.

## Requirements

- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
