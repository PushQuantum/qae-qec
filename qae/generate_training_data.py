from dataclasses import dataclass
import pickle
from dataclasses import field, dataclass, asdict
from dacite import from_dict

@dataclass
class TrainingData:
    qnnArch: list
    minibatch_size: int
    num_epochs: int
    batch_size: int # num of traning pairs
    ep: float # step size
    lda: float # inverse learning rate
    basis_states: list = field(default_factory=lambda : [])
    training_pairs: list = field(init=False)

    def make_training_dataset(self,basis_states = []):
        """Function to make to make traning data set
        arguments: 
            basis_states : The set of states used for training
        """
        if not self.basis_states:
            if not basis_states:
                raise("Please give the basis states to make the traning pair")
            self.basis_states = basis_states
        repeat_count = int(self.batch_size/len(self.basis_states))
        self.training_pairs =  self.basis_states * repeat_count
        

    def pickle_training_data_instance(self):
        with open(f"assets/{self.qnnArch}.pkl", "wb") as f:
            pickle.dump(asdict(self), f)
    
    @staticmethod
    def load_pickled_instace(path_to_pickle):
        with open(path_to_pickle, "rb") as f:
            dataDict =  pickle.load(f)
        
        return from_dict(TrainingData, dataDict)