from dataclasses import dataclass
import pickle
from dataclasses import field, dataclass, asdict
from dacite import from_dict
import json
from qutip import file_data_store, file_data_read, Qobj
import numpy as np
from cirq.ops import BitFlipChannel, PhaseFlipChannel, DepolarizingChannel

@dataclass
class TrainingData:
    """
    Class to represent the data that will be used for the quantum autoencoder training
    ...
    Attributes
    ----------
    qnnArch: list
        list representing the perceptron architecture of the DQNN
    minibatch_size: int
        size of the mini batch
    num_epochs: int
        the number of epochs to make the training for
    batch_size: int
        Total number of training pairs
    ep: float
        step size
    lda: float
        lambda representing lagrange multiplier?
    noise_channel: str
        the noise channel the training states will be subject to that noise channel
        options: "depolorizing", "bitflip", "phaseflip", "erasure"
    """
    qnnArch: list  # [3,1,3], [1,4,1]
    minibatch_size: int
    num_epochs: int
    batch_size: int  # num of training pairs
    ep: float  # step size
    lda: float # inverse learning rate
    num_basis_states: int
    noise_channel: str = "depolarizing"
    basis_states: list = field(default_factory=lambda: [])
    training_pairs: list = field(init=False)


    def __post_init__(self):
        if not self.basis_states:
            raise (
                "Please initialize the data set the basis states from which the dataset will be made."
            )
        self.num_basis_states = len(self.basis_states)
        self.make_training_dataset()

    def make_training_dataset(self):
        """Function to make training data set
        arguments:
            basis_states : The set of states used for training
        """
        # if not self.num_basis_states == len(self.basis_states):
        #     self.num_basis_states = len(self.basis_states)
        repeat_count = int(self.batch_size / len(self.basis_states))
        self.training_pairs = self.basis_states * repeat_count

    def make_noised_training_dataset(self):
        """
        Function to make noised dataset for training.
        """
        # if self.num_basis_states == len(self.basis_states):
        #     self.num_basis_states = len(self.basis_states)


    def save_as_json_and_dat(self):
        obj_dict = self.__dict__
        with open(f"assets/{str(self.qnnArch).replace(', ','_')}_hyperparams.json", "w") as outfile:
            json.dump({key: obj_dict[key] for key in list(obj_dict.keys())[:-2]}, outfile)
        file_data_store(
            f"assets/{str(self.qnnArch).replace(', ','_')}_basis_states.dat",
            np.vstack(self.basis_states),
            sep=",",
        )

    @staticmethod
    def load_from_json_and_dat(path_to_json, path_to_dat):
        with open(path_to_json, "r") as infile:
            obj_dict = json.load(infile)
        basis_states = [
            Qobj(state)
            for state in np.vsplit(
                file_data_read(path_to_dat, sep=","), obj_dict["num_basis_states"]
            )
        ]
        # objDict["num_basis_states"] = len(basis_states)
        obj_dict["basis_states"] = basis_states
        return from_dict(TrainingData, obj_dict)

    def pickle_training_data_instance(self):
        with open(f'assets/{str(self.qnnArch).replace(", ","_")}.pkl', "wb") as f:
            pickle.dump(asdict(self), f)

    @staticmethod
    def load_pickled_instance(path_to_pickle):
        with open(path_to_pickle, "rb") as f:
            data_dict = pickle.load(f)

        return from_dict(TrainingData, data_dict)
