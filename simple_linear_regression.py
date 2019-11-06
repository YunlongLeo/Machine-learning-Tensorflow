from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
rng = np.random

attr = input("input the attribute you want to use (eg. X1) : ")

# Parameters
learning_rate = 0.01
training_epochs = 100
display_step = 10

randNum=rng.randn()

# Read File
trainXData=[]
trainYData=[]
testXData=[]
testYData=[]

dataset_path = "data/Real estate valuation data set.xlsx"

column_names = ['X1','X2','X3','X4','X5','X6','Y']
raw_dataset = pd.read_excel(dataset_path,index_col=0,names=column_names,comment='#')
dataset = raw_dataset.copy()
dataset.tail()

train_dataset = dataset.sample(frac=0.9,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x-train_stats['mean']) / train_stats['std']

train_dataset = norm(train_dataset)
test_dataset = norm(test_dataset)

trainXData = train_dataset.get(attr).values
trainYData = train_dataset.get('Y').values
testXData = test_dataset.get(attr).values
testYData = test_dataset.get('Y').values



# Training Data
train_X = np.asarray(trainXData)
train_Y = np.asarray(trainYData)
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.multiply(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Testing example, as requested (Issue #2)
    test_X = np.asarray(testXData)
    test_Y = np.asarray(testYData)
    
    count = 0
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        
        #Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b))


    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    print ("Testing... (L2 loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation-Y, 2))/(2*test_X.shape[0]),
                        feed_dict={X: test_X, Y: test_Y}) #same function as cost above
    print ("Testing cost=", testing_cost)
    print ("Absolute l2 loss difference:", abs(training_cost - testing_cost))
