from numpy import sqrt
from generate_training_data import *
import qutip as qt
from DQNN import qubit0, qubit1 


qubit_plus = (1/sqrt(2)) * (qubit0 + qubit1)
basis_states = list()
basis_states.append(qt.tensor(qt.tensor(qubit0, qubit0), qubit0))
basis_states.append(qt.tensor(qt.tensor(qubit1, qubit1), qubit1))
basis_states.append(qt.tensor(qt.tensor(qubit_plus, qubit_plus), qubit_plus))

trInstance = TrainingData(qnnArch=[3,1,3], minibatch_size=3, num_epochs=200, batch_size=150,ep=0.1, lda=0.1, basis_states=basis_states)
trInstance.make_training_dataset()
trInstance.pickle_training_data_instance()