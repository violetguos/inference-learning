def linearMSE(y_hat, target):
    '''
    TODO: the MSE calculation
    '''
    print("y_hat",  y_hat)
    target = tf.cast(target, dtype = tf.float32)
    print("target", target)
    mse_mat = tf.square(tf.subtract(y_hat, target))
    print("msemst", mse_mat)
    loss = tf.reduce_mean(mse_mat)/2.0
    return loss



'''
Input: _data is x in the equation, dim by 784 flattened tensor
       _target is y in the equaion
       _regLambda is the wegithed decay coeff
       _learningRate is the epsilon
'''

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
#compute the current loss
mseCurr = linearMSE(y_hat, y)
#compute the decay/regularization term
regTerm = tf.multiply(_regLambda, tf.reduce_mean(tf.square(W)))
mseCurr = tf.add(mseCurr, regTerm)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = _learningRate)
train = optimizer.minimize(loss=meanSquaredError)

