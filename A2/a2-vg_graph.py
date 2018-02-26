#declare using a placeholder
W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.1))
W = tf.cast(W, dtype=tf.float32)
#initialize to a gaussian distr, honestly anything would work
b = tf.Variable(0.0, name='biases')
