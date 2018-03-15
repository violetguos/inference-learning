import tensorflow as tf
import numpy as np



# params
weight_decay = 0.01
training_epochs = 5000
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
		return trainData, trainTarget, validData, validTarget, testData, testTarget



trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
trainData = trainData.reshape(trainData.shape[0], 784)
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")


prediction = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
lD = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=prediction)) / num_samples

lW = weight_decay * tf.norm(W) / 2

# binary cross entropy function
cost = lD + lW

learning_rates = [0.005, 0.001, 0.0001]
losses = list()

for learning_rate in learning_rates:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
		loss=cost,
	)
	init = tf.global_variables_initializer()
	print ("batch size: " + str(batch_size))

	with tf.Session() as sess:
		sess.run(init)
		num_batches = int(trainData.shape[0] / batch_size)
		for epoch in range(training_epochs):
			
			c = None
			for i in range(num_batches):
				trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
				trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
				sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})
				if epoch % display_step == 0:
					c = sess.run(cost, feed_dict={X: trainBatchi, Y:trainTargeti})

			if epoch % display_step == 0:	
				print("Epoch: " + str(epoch) + ", cost: " + str(c))

		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		print("Train cost: " + str(train_loss))
		losses.append(train_loss)

print (losses)