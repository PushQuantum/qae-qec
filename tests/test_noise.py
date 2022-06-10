# import sys
import numpy as np
# import sys
# sys.path.append("/home/varunseshadri/projects/qosf-mentorship-cohort-5/qae_qec/")
# print(sys.path)
from qae_qec.qae.noise import *
# import qutip as qt

noise_model = Noise(noise_type="bit-flip",error_probability=0.1)

mixed_state = noise_model.prepare_noised_dataset(state="0_L",num_qubits=3)
print(mixed_state)
