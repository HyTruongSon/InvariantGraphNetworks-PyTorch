# Invariant and  Equivariant Graph Networks (PyTorch)
A PyTorch implementation of The ICLR 2019 paper "Invariant and  Equivariant Graph Networks" by Haggai Maron, Heli Ben-Hamu, Nadav Shamir and Yaron Lipman
https://openreview.net/forum?id=Syx72jC9tm. The official TensorFlow implementation is at https://github.com/Haggaim/InvariantGraphNetworks


## Data
Data should be downloaded from: https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?dl=0. 
Run the following commands in order to unzip the data and put its proper path.
```
mkdir data
unzip benchmark_graphs.zip -d data
```

### Prerequisites

Python3

PyTorch 1.5.0

Additional modules: numpy, pandas, matplotlib

TensorFlow is not neccessary except if you want to run the tests (comparisons) between the PyTorch and TensorFlow versions.


### Running the tests

Run the tests comparing between PyTorch and TensorFlow versions' tensor contractions. All tensor contractions are implemented 1-to-1. The two versions have identical tensor contractions:
```
cd layers/
python3 test_tensorflow_pytorch_contractions.py
```
Run the (permutation) equivariance tests for the equivariant linear layers implemented in PyTorch (e.g. permute the input tensor and the output tensor must transform covariantly):
```
python3 test_pytorch_layers.py
```

### Running the experiment

The folder main_scripts contains scripts that run different experiments:

1. To run 10-fold cross-validation with our hyper parameters run the main_10fold_experiment.py script. You can choose the datase in 10fold_config.json.
2. To run hyper-parameter search, run the main_parameter_search.py script  with the corresponding config file
3. To run training and evaluation on one of the data sets run the main.py script

example to run 10-fold cross-validation experiment:

```
cd main_scripts/
python3 -m main_10fold_experiment --config=../configs/10fold_config.json
```

### Note

PyTorch implementation of tensor contractions and equivariant linear layers is in:
```
layers/equivariant_linear_pytorch.py
```
PyTorch implementation of invariant (basic) graph nets:
```
models/invariant_basic.py
```


### Related work

Covariant Compositional Networks For Learning Graphs https://arxiv.org/abs/1801.02144
Predicting molecular properties with covariant compositional networks https://aip.scitation.org/doi/10.1063/1.5024797
The general theory of permutation equivarant neural networks and higher order graph variational encoders https://arxiv.org/abs/2004.03990

### Contact

Email: hytruongson@uchicago.edu


