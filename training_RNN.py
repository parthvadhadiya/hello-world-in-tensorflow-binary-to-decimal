import create_dataset as ds
import tensorflow as tf
from tensorflow.contrib import rnn	
import numpy as np

model_path = 'model'
x_len = 8
no_of_samples = 20000
lr = 0.01
training_steps= 20000

n_input = x_len
timestep = 1
n_hidden = 16
n_output = 1

trainX, trainY = ds.datasets(no_of_samples, x_len)
testX, testY = ds.datasets(20, x_len)

X = tf.placeholder(tf.float32, [None, timestep, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])


weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
bias = tf.Variable(tf.random_normal([n_output]))

def RNN(x, W, b):
	x = tf.unstack(x, timestep, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], W) + b

logits = RNN(X, weights, bias)

trainX = np.reshape(trainX, [-1, timestep, n_input])
trainY = np.reshape(trainY, [-1, n_output])

testX = np.reshape(testX, [-1, timestep, n_input])
testY = np.reshape(testY, [-1, n_output])

loss = tf.reduce_mean(tf.losses.mean_squared_error(logits, Y))
optimizer = tf.train.RMSPropOptimizer(lr)
train = optimizer.minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	
	for step in range(training_steps):
		_, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
		print("Step: ", step+1, "\tLoss: ", _loss)

	print("Optimization Finished")

	save_path = saver.save(sess, 'my_model')
	result = sess.run(logits, feed_dict={X: testX})
	result = sess.run(tf.round(result))

	print("Real \t Guess")
	for i in range(20):
		print(testY[i], ' === ', result[i] )
