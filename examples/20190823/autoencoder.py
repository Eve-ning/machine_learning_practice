# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:41:47 2019

@author: johnc
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

data_x = np.random.normal(0, 1, [1000, 3])
data_y = data_x

var_input = tf.placeholder(dtype=tf.float32, shape=[1,3])

var_w1 = tf.Variable(tf.truncated_normal(shape=[3,20]), dtype=tf.float32)
var_b1 = tf.Variable(tf.zeros([20]), dtype=tf.float32)
var_layer1 = \
  tf.nn.relu(tf.matmul(var_input, var_w1) + var_b1)

var_w2 = tf.Variable(tf.truncated_normal(shape=[20,3]), dtype=tf.float32)
var_b2 = tf.Variable(tf.zeros([3]), dtype=tf.float32)
var_output = \
  tf.matmul(var_layer1, var_w2) + var_b2
  
var_expected = tf.placeholder(dtype=tf.float32, shape=[1,3])

var_loss = tf.reduce_mean(tf.square(var_expected - var_output))
var_loss_op = tf.train.AdamOptimizer(0.001).minimize(var_loss)

epochs = 10
loss_list = []
loss_current = 1000000

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  step = 0
  for epoch in range(epochs):
    for x, y in zip(data_x, data_y):
      _, loss = sess.run([var_loss_op, var_loss],
                         feed_dict = {
                             var_input: [x],
                             var_expected: [x]
                             })
      loss_list.append(loss)
      if (loss < loss_current):
          print("Step {}\tNew Low Loss {}".format(step, loss))
          loss_current = loss
      
      step += 1
  
plt.scatter(range(10000), loss_list)
