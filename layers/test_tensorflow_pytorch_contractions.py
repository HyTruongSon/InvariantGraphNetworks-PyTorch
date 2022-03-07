# Numpy
import numpy as np

# TensorFlow
import tensorflow as tf

from equivariant_linear_tensorflow import ops_1_to_1, ops_1_to_2, ops_2_to_1, ops_2_to_2

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from equivariant_linear_pytorch import contractions_1_to_1, contractions_1_to_2, contractions_2_to_1, contractions_2_to_2

# Tensor comparison
def compare(tf_result, torch_result):
	eps = 1e-4
	if len(tf_result) != len(torch_result):
		return False
	for i in range(len(tf_result)):
		tf_tensor = tf_result[i]
		torch_tensor = torch_result[i].detach().numpy()
		error = np.sum(abs(tf_tensor - torch_tensor))
		if error > eps:
			print(i)
			print(error)
			return False
	return True

# Tensorflow session
sess = tf.InteractiveSession()

# Dimensions
N = 20 # Batch size
D = 10 # Number of vertices
m = 5 # Number of channels
dim = 4

# +------------+
# | ops_1_to_1 |
# +------------+

# Numpy
value = np.random.rand(N, D, m)

# TensorFlow
inputs = tf.convert_to_tensor(value = value, dtype = tf.float32)
ops = ops_1_to_1(inputs, dim = dim)
with sess.as_default():
	tf_result = sess.run(ops)
print("Done running ops_1_to_1 (tensorflow)")

# PyTorch
inputs = inputs.eval(session = tf.compat.v1.Session()) # To numpy array
inputs = torch.tensor(inputs) # To torch tensor
torch_result = contractions_1_to_1(inputs, dim = dim)
print("Done running contractions_1_to_1 (pytorch)")

assert compare(tf_result, torch_result) == True

# +------------+
# | ops_1_to_2 |
# +------------+

# Numpy
value = np.random.rand(N, D, m)

# TensorFlow
inputs = tf.convert_to_tensor(value = value, dtype = tf.float32)
ops = ops_1_to_2(inputs, dim = dim)
with sess.as_default():
	tf_result = sess.run(ops)
print("Done running ops_1_to_2 (tensorflow)")

# PyTorch
inputs = inputs.eval(session = tf.compat.v1.Session()) # To numpy array
inputs = torch.tensor(inputs) # To torch tensor
torch_result = contractions_1_to_2(inputs, dim = dim)
print("Done running contractions_1_to_2 (pytorch)")

assert compare(tf_result, torch_result) == True

# +------------+
# | ops_2_to_1 |
# +------------+

# Numpy
value = np.random.rand(N, D, m, m)

# TensorFlow
inputs = tf.convert_to_tensor(value = value, dtype = tf.float32)
ops = ops_2_to_1(inputs, dim = dim)
with sess.as_default():
	tf_result = sess.run(ops)
print("Done running ops_2_to_1 (tensorflow)")

# PyTorch
inputs = inputs.eval(session = tf.compat.v1.Session()) # To numpy array
inputs = torch.tensor(inputs) # To torch tensor
torch_result = contractions_2_to_1(inputs, dim = dim)
print("Done running contractions_2_to_1 (pytorch)")

assert compare(tf_result, torch_result) == True

# +------------+
# | ops_2_to_2 |
# +------------+

# Numpy
value = np.random.rand(N, D, m, m)

# TensorFlow
inputs = tf.convert_to_tensor(value = value, dtype = tf.float32)
ops = ops_2_to_2(inputs, dim = dim)
with sess.as_default():
	tf_result = sess.run(ops)
print("Done running ops_2_to_2 (tensorflow)")

# PyTorch
inputs = inputs.eval(session = tf.compat.v1.Session()) # To numpy array
inputs = torch.tensor(inputs) # To torch tensor
torch_result = contractions_2_to_2(inputs, dim = dim)
print("Done running contractions_2_to_2 (pytorch)")

assert compare(tf_result, torch_result) == True

print("Done all tests successfully!")
