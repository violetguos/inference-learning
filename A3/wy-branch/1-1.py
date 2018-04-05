import tensorflow as tf

#1-1


# vectorized layer
def get_layer(input_tensor, num_hidden_units):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	num_input_units = input_tensor.shape[0]
	w = tf.get_variable(name="weights", shape=(num_input_units, num_hidden_units), dtype=tf.float32, initializer=weight_initializer)
	b = tf.Variable(tf.zeros(shape=(num_hidden_units)), name="bias")
	z = tf.add(tf.matmul(input_tensor), b)
	return z
