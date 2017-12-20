import create_dataset as ds
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
x_len = 8
no_of_samples = 10000
lr = 0.01
training_steps= 5000


n_input = x_len
timestep = 1
n_hidden = 16
n_output = 1

testX = [[ 1,  0,  1,  1,  1,  0,  0,  1,]]

print(testX , "after")
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

testX = np.reshape(testX, [-1, timestep, n_input])

print(testX)
saver = tf.train.Saver()

with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state('./')
	saver.restore(sess, ckpt.model_checkpoint_path)
	result = sess.run(logits, feed_dict={X: testX})
	result = sess.run(tf.round(result))
	print("result is ",result)