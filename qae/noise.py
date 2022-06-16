from typing import List, Union, Optional, Tuple
from collections import namedtuple
import numpy as np
# import qutip as qt
import cirq
from dataclasses import  dataclass, field, asdict
import pandas as pd
import pickle
import pathlib
from datetime import date
from utils import pure_logical_plus, pure_logical_one, pure_logical_zero
import json

INIT_SEED = 123124
default_rng = np.random.default_rng(seed= INIT_SEED)

sigmaX = np.array([[0., 1.],[1., 0.]], dtype=object)
sigmaY = np.array([[0. -1.j],[1.j, 0.j]], dtype=object)
sigmaZ = np.array([[1., 0.],[0, 1.]], dtype=object)

LOGICAL_STATES = ['0_L','1_L','+_L']

TrainingPair = namedtuple("TrainingPair","state noised_state pure_state") # used namedtuple for more readable code.

def prepare_pure_logical_state(state:str, num_qubits: int) -> np.ndarray:
    if state == "0_L":
        return pure_logical_zero(num_qubits)
    elif state == "1_L":
        return pure_logical_one(num_qubits)
    elif state == "+_L":
        return pure_logical_plus(num_qubits)
    else:
        raise "given logical state cannot be prepared yet!"

def prepare_noised_logical_state(state:str, qubits:List) -> Union["cirq.Moment", "cirq.Circuit"]:
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
    num_qubits: int = field(init= False)
    training_pairs: tuple = field(default_factory=lambda: ()) #[...,(state_label, noised_dm, pure_dm),...]

    def __post_init__(self):
        allowed_noise_channels = ["bit-flip", "phase-flip", "depolarizing", "erasure"]
        if self.noise_type not in allowed_noise_channels:
            raise "The allowed noise channels are bit-flip, phase-flip, depolarizing and erasure"
        assert 0.0 <= self.error_probability <=1.0, "The error probability should be between 0 and 1!"
        self.num_qubits = None

    def _apply_bit_flip_noise(self,qubits: List) -> "cirq.Moment":
        bit_flip = cirq.bit_flip(p = self.error_probability)
        return cirq.Moment([bit_flip.on_each(qubits)])

    def _apply_phase_flip_noise(self, qubits: List) -> "cirq.Moment":
        phase_flip = cirq.phase_flip(p = self.error_probability)
        return cirq.Moment([phase_flip.on_each(qubits)])

    def _apply_depolarizing_noise(self, qubits: List) -> "cirq.Moment":
        depolarize = cirq.depolarize(p = self.error_probability)
        return cirq.Moment([depolarize.on_each(qubits)])

    def prepare_noised_density_matrix(self, state: str, qubits: List["cirq.ops.Qid"]=None):
        """Apply noise channel to a given state and sample noised states from it.
        arguments:
            state: quantum state that noise must be applied to, must be a density operator
        """
        qubits = cirq.LineQubit.range(3) if qubits is None else qubits
        circuit = cirq.Circuit()
        circuit.append(prepare_noised_logical_state(state, qubits))
        fn_to_apply = getattr(self, F"_apply_{self.noise_type.replace('-','_')}_noise")
        circuit.append(fn_to_apply(qubits))
        circuit.append(cirq.Moment([cirq.measure(qubit) for qubit in qubits]))
        return sample_circuit(qubits, circuit)

    def prepare_noised_training_pair(self, num_qubits: int, logical_states:List[str]=None, num_training_pairs: int = 50) -> List[Tuple]:
        """
        Given the logical_states to sample and the number of qubits, returns list of the density matrices.
        Each dm corresponds to a mixed state that is sampled in the given noise channel
        """

        self.num_qubits = num_qubits
        logical_states = LOGICAL_STATES if logical_states is None else logical_states
        qubits = cirq.LineQubit.range(num_qubits)
        self.training_pairs = [TrainingPair(state, self.prepare_noised_density_matrix(state, qubits), prepare_pure_logical_state(state, num_qubits)) for state in logical_states for _ in range(num_training_pairs)]
        return self.training_pairs

    def save_dataset(self, file_folder:str = None, file_name: str = None):
        """

        @param file_folder: The path of the folder in which the files will go
        @param file_name: The file name for the pickle
        @return: filename.pkl, that stores the Noise object, filename.json, that stores the hyoerparameters in a humaa readable form.
        """
        if file_name is None:
            file_name = f"{self.num_qubits}_{self.noise_type}_{str(self.error_probability).replace('.','-')}_{len(self.training_pairs)}"
        if file_folder is None:
            file_folder = f"assets/{str(date.today()).replace('-','_')}"
            pathlib.Path(file_folder).mkdir(parents = True, exist_ok = True)
        hp_keys = ['num_qubits','noise_type','error_probability']
        hp = dict()
        hp['pickle_path'] = f"{file_folder}/{file_name}.pkl"
        for key in hp_keys:
            hp[key] = self.__dict__[key]
        print(hp)
        with open(f"{file_folder}/{file_name}.pkl","wb") as f:
            pickle.dump(self, f)
        with open(f"{file_folder}/{file_name}.json","w") as f:
            json.dump(hp, f)

    @staticmethod
    def load_dataset(path_to_json: str):
        """
        input: json which containts the path to the pickle.
        returns : Noise object
        """
        with open(path_to_json, "r") as f:
            hp  = json.load(f)
            pkl_path = hp['pickle_path']
        with open(pkl_path,"rb") as obj:
            return pickle.load(obj)


class NoiseSweep:
    allowed_params = ('error_probability', 'noise_type')

    def __init__(self, swept_param: str, **kwargs):
        """
        kwargs:
            attributes of class Noise
        """
        self.swept_param = swept_param
        self.noise_models = [Noise(error_probability=kwargs['error_probability'], noise_type=kwargs['noise_type'])]
        swept_dataset: pd.DataFrame
        if 'dataset' in kwargs.keys():
            self._set_dataset_to_each_noise_model()



    def _set_dataset_to_each_noise_model(self):
        raise NotImplementedError

    def prepare_database(self,num_qubits: int, logical_states: List[str], **kwargs):
        """

        @param num_qubits: Number qubits in each logical state.
        @param logical_states: The logical states that will be fed to the autoencoder as training states
        @param kwargs: swept_param, sweeper
        @return:
        """
        logical_states = LOGICAL_STATES if logical_states is None else logical_states
        for nm in self.noise_models:
            nm.prepare_noised_dataset(num_qubits,logical_states, num_training_pairs=50)


    def save_database(self):
        NotImplementedError

    @staticmethod
    def load_database(path_to_database):
        NotImplementedError
