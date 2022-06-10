import numpy as np
from enum import Enum

def pure_logical_zero(num_qubits:int) -> np.ndarray:
    arr = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex)
    arr[0][0] = 1. + 0.j
    return arr
def pure_logical_one(num_qubits:int) -> np.ndarray:
    arr = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex)
    arr[-1][-1] = 1. + 0.j
    return arr
def pure_logical_plus(num_qubits:int) -> np.ndarray:
    arr = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex)
    arr[0][0] = 0.5 + 0.j
    arr[-1][-1] = 0.5 + 0.j
    return arr

