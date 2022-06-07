from qae.generate_training_data import TrainingData

path_to_json = "/home/varunseshadri/projects/qosf-mentorship-cohort-5/qae_qec/qae/assets/[3_1_3]_hyperparams.json"
path_to_dat = "/home/varunseshadri/projects/qosf-mentorship-cohort-5/qae_qec/qae/assets/[3_1_3]_basis_states.dat"

test = TrainingData.load_from_json_and_dat(path_to_json, path_to_dat)

assert len(test.training_pairs) == 150
