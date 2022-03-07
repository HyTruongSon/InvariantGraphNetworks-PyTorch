# Numpy
import numpy as np

# TensorFlow
import tensorflow as tf

from layers_tensorflow import diag_offdiag_maxpool

# Tensorflow session
sess = tf.InteractiveSession()

# Dimensions
N = 20 # Batch size
D = 10 # Number of vertices
m = 5 # Number of channels

# Numpy
value = np.random.rand(N, D, m, m)

# TensorFlow
inputs = tf.convert_to_tensor(value = value, dtype = tf.float32)
ops = diag_offdiag_maxpool(inputs)
with sess.as_default():
	tf_result = sess.run(ops)
print("Done running tensorflow")

print("Done all tests successfully!")
