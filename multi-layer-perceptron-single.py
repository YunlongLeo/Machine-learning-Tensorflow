from __future__ import print_function

import tensorflow as tf
import scipy.io
import numpy as np
import xlrd
import random

datasetX = [[0]*8]*27
datasetY = [[0]*8]*27

attr = input("Please input your file path (data/jumping jacks sets.xlsx):")
axis = input("Please input your axis (x/y/z):")
wb = xlrd.open_workbook(attr)
sheet = wb.sheet_by_index(0)
for i in range(sheet.ncols):
    if axis == 'x':
        datasetX[i] = sheet.col_values(i)[0:8]
    elif axis == 'y':
        datasetX[i] = sheet.col_values(i)[8:16]
    elif axis == 'z':
        datasetX[i] = sheet.col_values(i)[16:24]
    else:
        print("invalid input")
    datasetY[i] = sheet.col_values(i)[24:30]

random.shuffle(datasetX)
random.shuffle(datasetY)

test_size = int(len(datasetX) * 0.1 +1)
test_dataX = datasetX[:test_size]
train_dataX = datasetX[test_size:]
test_dataY = datasetY[:test_size]
train_dataY = datasetY[test_size:]

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 1
display_step = 10

# Network Parameters
n_hidden_1 = 10  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
n_hidden_3 = 10  # 3rd layer number of neurons
n_input = len(train_dataX[0]) # MNIST data input (img shape: 1*8)
n_classes = 6  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 1
        # Loop over all batches
        for i in range(0,total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: train_dataX,
                                                            Y: train_dataY})
            # Compute average loss
            avg_cost += c/45
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test_dataX, Y: test_dataY}))
