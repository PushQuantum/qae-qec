import numpy as np
from scipy.stats import unitary_group

def generate_random_square_unitaries(input_qubit_size: int, output_qubit_size:int):
    return unitary_group.rvs(dim =2**(max(input_qubit_size, output_qubit_size)) )

def verify_unitarity(unitary: np.ndarray):
    return np.allclose(np.dot(unitary, unitary.conj().T), np.eye(np.shape(unitary)[0]))

def brownian_circuit():
    pass

def random_unitary():
    