
import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.005
training_epochs = 20000
display_step = 50
batch_size = 500



print ("hello world")
def get_data():
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
		return trainData, trainTarget, validTarget, validTarget, testData, testTarget


trainData, trainTarget, validTarget, validTarget, testData, testTarget = get_data()
trainData = trainData.reshape(trainData.shape[0], 784)
n_samples = trainData.shape[0]
print (trainData.shape)
print (trainTarget.shape)

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))

b = tf.Variable(tf.ones(1), name="bias")
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), Y)

pred = tf.add(tf.matmul(X, W), b)
cost = tf.reduce_sum(tf.norm(pred - Y)) / (2*n_samples)


with tf.Session() as sess:

	init = tf.global_variables_initializer()
	sess.run(init)
	loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
	print (loss)