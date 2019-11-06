from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 100
display_step = 10

randNum=rng.randn()

# Read File
trainX1Data=[]
trainX2Data=[]
trainX3Data=[]
trainX4Data=[]
trainX5Data=[]
trainX6Data=[]
trainYData=[]

testX1Data=[]
testX2Data=[]
testX3Data=[]
testX4Data=[]
testX5Data=[]
testX6Data=[]
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

trainX1Data = train_dataset.get('X1').values
trainX2Data = train_dataset.get('X2').values
trainX3Data = train_dataset.get('X3').values
trainX4Data = train_dataset.get('X4').values
trainX5Data = train_dataset.get('X5').values
trainX6Data = train_dataset.get('X6').values
trainYData = train_dataset.get('Y').values

testX1Data = test_dataset.get('X1').values
testX2Data = test_dataset.get('X2').values
testX3Data = test_dataset.get('X3').values
testX4Data = test_dataset.get('X4').values
testX5Data = test_dataset.get('X5').values
testX6Data = test_dataset.get('X6').values
testYData = test_dataset.get('Y').values




# Training Data
train_X1 = np.asarray(trainX1Data)
train_X2 = np.asarray(trainX2Data)
train_X3 = np.asarray(trainX3Data)
train_X4 = np.asarray(trainX4Data)
train_X5 = np.asarray(trainX5Data)
train_X6 = np.asarray(trainX6Data)
train_Y = np.asarray(trainYData)
n_samples = train_X1.shape[0]

# tf Graph Input
X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
X3 = tf.placeholder("float")
X4 = tf.placeholder("float")
X5 = tf.placeholder("float")
X6 = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W1 = tf.Variable(rng.randn(), name="weight1")
W2 = tf.Variable(rng.randn(), name="weight2")
W3 = tf.Variable(rng.randn(), name="weight3")
W4 = tf.Variable(rng.randn(), name="weight4")
W5 = tf.Variable(rng.randn(), name="weight5")
W6 = tf.Variable(rng.randn(), name="weight6")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.multiply(X1, W1), b)
activation = tf.add(tf.multiply(X2, W2), activation)
activation = tf.add(tf.multiply(X3, W3), activation)
activation = tf.add(tf.multiply(X4, W4), activation)
activation = tf.add(tf.multiply(X5, W5), activation)
activation = tf.add(tf.multiply(X6, W6), activation)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Testing example, as requested (Issue #2)
    test_X1 = np.asarray(testX1Data)
    test_X2 = np.asarray(testX2Data)
    test_X3 = np.asarray(testX3Data)
    test_X4 = np.asarray(testX4Data)
    test_X5 = np.asarray(testX5Data)
    test_X6 = np.asarray(testX6Data)
    test_Y = np.asarray(testYData)
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x1,x2,x3,x4,x5,x6,y) in zip(train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_Y):
            sess.run(optimizer, feed_dict={X1: x1,X2: x2,X3: x3,X4: x4,X5: x5,X6: x6, Y: y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X1: train_X1,X2: train_X2,X3: train_X3,X4: train_X4,X5: train_X5,X6: train_X6, Y:train_Y})), \
            "W1=",sess.run(W1),"W2=",sess.run(W2),"W3=",sess.run(W3),"W4=",sess.run(W4),"W5=",sess.run(W5),"W6=", sess.run(W6), "b=", sess.run(b))


    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X1: train_X1,X2: train_X2,X3: train_X3,X4: train_X4,X5: train_X5,X6: train_X6,Y: train_Y})
    print ("Training cost=", training_cost, "W1=", sess.run(W1),"W2=", sess.run(W2),"W3=", sess.run(W3),"W4=", sess.run(W4),"W5=", sess.run(W5),"W6=", sess.run(W6), "b=", sess.run(b), '\n')

    print ("Testing... (L2 loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation-Y, 2))/(2*test_X1.shape[0]),
                        feed_dict={X1: test_X1,X2: test_X2,X3: test_X3,X4: test_X4,X5: test_X5,X6: test_X6, Y: test_Y}) #same function as cost above
    print ("Testing cost=", testing_cost)
    print ("Absolute l2 loss difference:", abs(training_cost - testing_cost))
