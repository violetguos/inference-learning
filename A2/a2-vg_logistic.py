'''
function to build a logistic graph for training

input param:
  _data: x
  _target: y
  _learningRate: learning_rate
  _regLambda: regularizer term

output param:
  W
  x
  y
  train, optimizer
'''

#initializxe using placeholders
#copied from linear
#declare using a placeholder, feed in _data and _target to x ,y 
x_dim, dum1 =_data.get_shape().as_list()
x = tf.placeholder(tf.float32, shape=[x_dim, 784], name='dataX')
# W initialize to a gaussian distr, honestly anything would work
W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.1), name='weights')
W = tf.cast(W, dtype=tf.float32)
b = tf.Variable(0.0, name='biases')
y = tf.placeholder(tf.float32, shape=[x_dim, 1], name='targetY')


#compute the current y_hat
y_hat =  tf.matmul(x, W) + b
regTerm = tf.multiply(_regLambda, tf.reduce_mean(tf.square(W)))

#compute crossEntropyError
crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits= y_hat)
crossEntropyError = tf.reduce_mean(crossEntropy)/2.0 + regTerm/2.0

optimizer = tf.train.AdamOptimizer(learning_rate = _learningRate)
train = optimizer.minimize(loss=crossEntropyError)

