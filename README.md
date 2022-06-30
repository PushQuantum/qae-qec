# Quantum Autoencoders for Quantum Error Correction
This a project aimed at replicating the results from the arXiV paper on using [Quantum Autoencoders for Quantum Error Correction](https://arxiv.org/abs/2202.00555)(qae-paper). The results in this paper was achieved by using DQNNs ([Disspiative Quantum Neural Networks](https://github.com/qigitphannover/DeepQuantumNeuralNetworks)) with code written in MATLAB and they are numerical simulation results. 

Desired Objectives for this project:

- [x] Replicate the results but using Python. Insiped from [DQNN Repo](https://github.com/qigitphannover/DeepQuantumNeuralNetworks/tree/master/Autoencoder-MATLAB)
  - [x] Make datasets and dataset generation code public.     
- [ ] Run the learned encodings and decodings on actual hardware.        
- [ ] Try gradient free methods for neural network training, trying an alternate solution for the Barren Plateau problem.

https://1drv.ms/p/s!AhzKZHA1xnhDiK9wIC2fFvj9QZDEKg?e=LzqNnt - Short presentation explaning the theory of the paper.


------------------------------------------
What is new in this repo, compared to the autoencoder implementation of [DQNN Repo](https://github.com/qigitphannover/DeepQuantumNeuralNetworks/tree/master/Autoencoder-MATLAB). 
  1. methods to perform training for autoenoders in self inverse architecture of the autoencoders from qae-paper.
  2. datasets for training available as pandas DataFrames
  3. Circuit Implementation of the autoencoder as a VQA, with aim of making it hardware compatible.
  
