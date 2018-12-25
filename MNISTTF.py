#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:22:16 2018

@author: shashi
"""
import tensorflow as tf
import numpy as np
import struct
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#parameters
num_epochs = 200
mini_size = 10000
learning_rate = 0.02
ops.reset_default_graph()
with open('MNIST/traindata', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
data = data.reshape((size, nrows, ncols))
X_train = (data.reshape(data.shape[0], -1).T)/255
#X_train = X_train[:,:10000]
X = tf.placeholder(tf.float32,[X_train.shape[0], None],name="X")

with open('MNIST/testdata', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
data = data.reshape((size, nrows, ncols))
X_test = (data.reshape(data.shape[0], -1).T)/255

with open('MNIST/trainlabel', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
y_train = np.array(data,ndmin = 1)
#y_train = y_train[:10000]
y_train = tf.one_hot(y_train, 10, axis = 0)

with open('MNIST/testlabel', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
y_test = np.array(data,ndmin = 1)
y_test = tf.one_hot(y_test, 10, axis = 0)
with tf.Session() as session:
    y_train = session.run(y_train)
with tf.Session() as session:
    y_test = session.run(y_test)
Y = tf.placeholder(tf.float32,[y_train.shape[0], None],name="Y")
W1 = tf.get_variable("W1",[10, X_train.shape[0]],initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1",[10, 1],initializer = tf.zeros_initializer())
Z1 = tf.add(tf.matmul(W1,X),b1)
A1 = tf.nn.relu(Z1)
W2 = tf.get_variable("W2",[y_train.shape[0], 10],initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2",[y_train.shape[0], 1], initializer = tf.zeros_initializer())
Z2 = tf.add(tf.matmul(W2,A1),b2)
parameters = {"W1":W1,
              "b1":b1,
              "W2":W2,
              "b2":b2}
logits = tf.transpose(Z2)
labels = tf.transpose(Y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
init = tf.global_variables_initializer()
costs = [];
with tf.Session() as session:
    session.run(init)
    for epoch in range(num_epochs):
        epoch_cost = 0. 
        num_mini = int(X_train.shape[1]/mini_size)
        for i in range(num_mini):
            mini_x = X_train[:,i*mini_size:i*mini_size+mini_size]
            mini_y = y_train[:,i*mini_size:i*mini_size+mini_size]
            _ , minibatch_cost = session.run([optimizer,cost], feed_dict = {X:mini_x, Y:mini_y})
        epoch_cost += minibatch_cost / num_mini
        if epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 10 == 0:
            costs.append(epoch_cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    # lets save the parameters in a variable
    parameters = session.run(parameters)
    print ("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))