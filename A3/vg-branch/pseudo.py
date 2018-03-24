'''
	this file contains pseudo code for A3
'''

# part 1 
class NeuralNetowork:

	def __init__():
		self.num_layer
		self.hidden_unit


#part 1.1 layer wise building bock
def build_layer(input_tensor, num_units):
	input_tensor # dim []

	'''
	input tensor
	[X_1  X_2 ]
	
	xavier
	[ W_1,1 
	  W_1,2 ]
	'''

	W = Xavier_init(num_input = X dim, output = 1 )
	tf.mult(input_tensor, W) #vectorized mulitplication
	
	
#part 1.2 build a NN and try learning rates
learningRateArr = [0.001, 0.01, 0.005]

def buildNetwork():
	# TODO: initialize nodes, connect weight edges
	# ANS: do so by init layer weight(x_size_prev, hidden_size)
	# Relu Layer
	layer_1 = tf.add(wtx, bias)
	layer_1 = tf.nn.relu(layer_1)
	layer_1 = tf.nn.dropout(layer_1, prob = 0.5)
	out_layer = wtx + bias

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


	

