from typing import List, Union

import numpy as np
# import qutip as qt
import cirq
from dataclasses import  dataclass, field

INIT_SEED = 123124
default_rng = np.random.default_rng(seed= INIT_SEED)

sigmaX = np.array([[0., 1.],[1., 0.]], dtype=object)
sigmaY = np.array([[0. -1.j],[1.j, 0.j]], dtype=object)
sigmaZ = np.array([[1., 0.],[0, 1.]], dtype=object)

def prepare_logical_state(state:str, qubits:List) -> Union["cirq.Moment", "cirq.Circuit"]:
    if state == "0_L":
        return cirq.Moment([])
    elif state == "1_L":
        return cirq.Moment([cirq.X.on(qubit) for qubit in qubits])
    elif state == "+_L":
        return cirq.Circuit([cirq.H.on(qubits[0]), [cirq.CNOT.on(control=qubits[0], target=qubit) for qubit in qubits[1:]]])
    else:
        raise "given logical state cannot be prepared yet!"

def sample_circuit(qubits: List['cirq.ops.Qid'], circuit: "cirq.Circuit", samples = 50, sampler = cirq.Simulator()) -> List[np.ndarray]:
    """
    Samples the circuit with a given noise model and returns a density matrix
    """
    result = sampler.run(circuit, repetitions=samples)
    states, probs = np.array([*result.multi_measurement_histogram(keys=qubits).keys()]), np.array([*result.multi_measurement_histogram(keys=qubits).values()])
    probs = probs/np.sum(probs)
    mixed_state = sum([p*cirq.to_valid_density_matrix(state, num_qubits=len(qubits)) for state, p in zip(states,probs)])
    return mixed_state


@dataclass
class Noise:
    noise_type: str
    error_probability: float = 0.1
    def __post_init__(self):
        allowed_noise_channels = ["bit-flip", "phase-flip", "depolarizing", "erasure"]
        if self.noise_type not in allowed_noise_channels:
            raise "The allowed noise channels are bit-flip, phase-flip, depolarizing and erasure"
        assert 0.0 <= self.error_probability <=1.0, "The error probability should be between 0 and 1!"


    def prepare_noised_dataset(self, state: "str", num_qubits: int) -> List["np.ndarray"]:
        """Apply noise channel to a given state and sample noised states from it.
        arguments:
            state: quantum state that noise must be applied to, must be a density operator
        """
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()
        circuit.append(prepare_logical_state(state, qubits))
        fn_to_apply = getattr(self, F"_apply_{self.noise_type.replace('-','_')}_noise")
        circuit.append(fn_to_apply(qubits))
        circuit.append(cirq.Moment([cirq.measure(qubit) for qubit in qubits]))
        return sample_circuit(qubits, circuit)

    def _apply_bit_flip_noise(self,qubits: List) -> "cirq.Moment":
        bit_flip = cirq.bit_flip(p = self.error_probability)
        return cirq.Moment([bit_flip.on_each(qubits)])

    def _apply_phase_flip_noise(self, qubits: List) -> "cirq.Moment":
        phase_flip = cirq.phase_flip(p = self.error_probability)
        return cirq.Moment([phase_flip.on_each(qubits)])

    def _apply_depolarizing_noise(self, qubits: List) -> "cirq.Moment":
        depolarize = cirq.depolarize(p = self.error_probability)
        return cirq.Moment([depolarize.on_each(qubits)])

    # def _apply_erasure_noise(self):
    #     pass



