# Numpy
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from equivariant_linear_pytorch import layer_2_to_2, layer_2_to_1, layer_1_to_2, layer_1_to_1, layer_basic

# Set common random seed
random_seed = 123456789
np.random.seed(random_seed)
torch.random.manual_seed(random_seed)

# Randomize permutation matrix
def perm_matrix(N, m):
	P = np.zeros((N, m, m))
	for n in range(N):
		perm = np.random.permutation(m)
		for i in range(m):
			j = perm[i] - 1
			P[n, i, j] = 1.0
	return P

# Permute tensor 1d
def permute_tensor_1d(T, P):
	P_transpose = P.transpose(1, 2)
	out = torch.einsum("bij,bdj->bdi", P_transpose, T)
	return out

# Permute tensor 2d
def permute_tensor_2d(T, P):
	P_transpose = P.transpose(1, 2)
	out = torch.einsum("bij,bdjk->bdik", P_transpose, T)
	out = torch.einsum("bdik,bkj->bdij", out, P)
	return out

# Equivariance permutation test for equi_2_to_2
def equivariance_test_2_to_2(layer, N, D, S, m):
	# Numpy
	value = np.random.rand(N, D, m, m)
	P = torch.tensor(perm_matrix(N, m), dtype = torch.float32)

	# PyTorch
	inputs = torch.tensor(value, dtype = torch.float32) # To torch tensor
	perm_inputs = permute_tensor_2d(inputs, P)

	outputs = layer(inputs)
	perm_outputs = layer(perm_inputs)

	err = torch.sum(torch.abs(permute_tensor_2d(outputs, P) - perm_outputs)).item()
	if err < 1e-3:
		return True
	return False

# Equivariance permutation test for equi_2_to_1
def equivariance_test_2_to_1(layer, N, D, S, m):
	# Numpy
	value = np.random.rand(N, D, m, m)
	P = torch.tensor(perm_matrix(N, m), dtype = torch.float32)

	# PyTorch
	inputs = torch.tensor(value, dtype = torch.float32) # To torch tensor
	perm_inputs = permute_tensor_2d(inputs, P)

	outputs = layer(inputs)
	perm_outputs = layer(perm_inputs)

	err = torch.sum(torch.abs(permute_tensor_1d(outputs, P) - perm_outputs)).item()
	if err < 1e-4:
		return True
	return False

# Equivariance permutation test for equi_1_to_2
def equivariance_test_1_to_2(layer, N, D, S, m):
	# Numpy
	value = np.random.rand(N, D, m)
	P = torch.tensor(perm_matrix(N, m), dtype = torch.float32)

	# PyTorch
	inputs = torch.tensor(value, dtype = torch.float32) # To torch tensor
	perm_inputs = permute_tensor_1d(inputs, P)

	outputs = layer(inputs)
	perm_outputs = layer(perm_inputs)

	err = torch.sum(torch.abs(permute_tensor_2d(outputs, P) - perm_outputs)).item()
	if err < 1e-4:
		return True
	return False

# Equivariance permutation test for equi_1_to_1
def equivariance_test_1_to_1(layer, N, D, S, m):
	# Numpy
	value = np.random.rand(N, D, m)
	P = torch.tensor(perm_matrix(N, m), dtype = torch.float32)

	# PyTorch
	inputs = torch.tensor(value, dtype = torch.float32) # To torch tensor
	perm_inputs = permute_tensor_1d(inputs, P)

	outputs = layer(inputs)
	perm_outputs = layer(perm_inputs)

	err = torch.sum(torch.abs(permute_tensor_1d(outputs, P) - perm_outputs)).item()
	if err < 1e-4:
		return True
	return False

# Equivariance permutation test for equi_basic
def equivariance_test_basic(layer, N, D, S, m):
	# Numpy
	value = np.random.rand(N, D, m, m)
	P = torch.tensor(perm_matrix(N, m), dtype = torch.float32)

	# PyTorch
	inputs = torch.tensor(value, dtype = torch.float32) # To torch tensor
	perm_inputs = permute_tensor_2d(inputs, P)

	outputs = layer(inputs)
	perm_outputs = layer(perm_inputs)

	err = torch.sum(torch.abs(permute_tensor_2d(outputs, P) - perm_outputs)).item()
	if err < 1e-4:
		return True
	return False

# Dimensions
N = 20 # Batch size
D = 10 # Number of input channels
S = 30 # Number of output channels
m = 5 # Number of vertices

# +-------------+
# | equi_2_to_2 |
# +-------------+

layer = layer_2_to_2(D, S)
assert equivariance_test_2_to_2(layer, N, D, S, m) == True
print("Done running layer_2_to_2 (pytorch)")

# +-------------+
# | equi_2_to_1 |
# +-------------+

layer = layer_2_to_1(D, S)
assert equivariance_test_2_to_1(layer, N, D, S, m) == True
print("Done running layer_2_to_1 (pytorch)")

# +-------------+
# | equi_1_to_2 |
# +-------------+

layer = layer_1_to_2(D, S)
assert equivariance_test_1_to_2(layer, N, D, S, m) == True
print("Done running layer_1_to_2 (pytorch)")

# +-------------+
# | equi_1_to_1 |
# +-------------+

layer = layer_1_to_1(D, S)
assert equivariance_test_1_to_1(layer, N, D, S, m) == True
print("Done running layer_1_to_1 (pytorch)")

# +------------+
# | equi_basic |
# +------------+

layer = layer_basic(D, S)
assert equivariance_test_basic(layer, N, D, S, m) == True
print("Done running layer_basic (pytorch)")

print("Done")
