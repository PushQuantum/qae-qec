from dataclasses import dataclass
import pickle
from dataclasses import field, dataclass, asdict
from dacite import from_dict
import json
from qutip import file_data_store, file_data_read, Qobj
import numpy as np


@dataclass
class TrainingData:
    qnnArch: list  # [3,1,3], [1,4,1]
    minibatch_size: int
    num_epochs: int
    batch_size: int  # num of traning pairs
    ep: float  # step size
    lda: float
    num_basis_states: int  # inverse learning rate
    basis_states: list = field(default_factory=lambda: [])
    training_pairs: list = field(init=False)

    def __post_init__(self):
        if not self.basis_states:
            raise (
                "Please initialize the data set the basis states from which the dataset will be made."
            )
        # self.num_basis_states = len(self.basis_states)
        self.make_training_dataset()

    def make_training_dataset(self):
        """Function to make to make traning data set
        arguments:
            basis_states : The set of states used for training
        """
        if self.num_basis_states == len(self.basis_states):
            self.num_basis_states = len(self.basis_states)
        repeat_count = int(self.batch_size / len(self.basis_states))
        self.training_pairs = self.basis_states * repeat_count

    def save_as_json_and_dat(self):
        objDict = self.__dict__
        with open(f"assets/{str(self.qnnArch).replace(', ','_')}_hyperparams.json", "w") as outfile:
            json.dump({key: objDict[key] for key in list(objDict.keys())[:-2]}, outfile)
        file_data_store(
            f"assets/{str(self.qnnArch).replace(', ','_')}_basis_states.dat",
            np.vstack(self.basis_states),
            sep=",",
        )

    @staticmethod
    def load_from_json_and_dat(path_to_json, path_to_dat):
        with open(path_to_json, "r") as infile:
            objDict = json.load(infile)
        basis_states = [
            Qobj(state)
            for state in np.vsplit(
                file_data_read(path_to_dat, sep=","), objDict["num_basis_states"]
            )
        ]
        # objDict["num_basis_states"] = len(basis_states)
        objDict["basis_states"] = basis_states
        return from_dict(TrainingData, objDict)

    def pickle_training_data_instance(self):
        with open(f'assets/{str(self.qnnArch).replace(", ","_")}.pkl', "wb") as f:
            pickle.dump(asdict(self), f)

    @staticmethod
    def load_pickled_instace(path_to_pickle):
        with open(path_to_pickle, "rb") as f:
            dataDict = pickle.load(f)

        return from_dict(TrainingData, dataDict)
