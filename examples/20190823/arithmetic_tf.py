# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:48:23 2019

@author: johnc
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

data_x_1 = np.random.normal(0, 1, [100, 1])
data_x_2 = np.random.normal(2, 1, [100, 1])
data_y = data_x_1 + data_x_2

var_input = tf.placeholder(tf.float32, shape=[2, 1],
                           name="input")
var_w1 = tf.Variable(initial_value=tf.truncated_normal([2,3]),
                     dtype=tf.float32,
                     expected_shape=[2,3])
var_b1 = tf.Variable(tf.zeros([3]))
var_output = tf.matmul(tf.transpose(var_input), var_w1) + var_b1

var_output_expected = tf.placeholder(tf.float32, shape=[1],
                                     name="output_expected")

var_loss = tf.reduce_mean(tf.square(var_output_expected - var_output))
var_loss_op = tf.train.AdamOptimizer(0.01).minimize(var_loss)

epochs = 100

loss_list = []

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(epochs):
    for x_1, x_2, y in zip(data_x_1, data_x_2, data_y):
      _, loss = sess.run([var_loss_op, var_loss],
                         feed_dict = {
                             var_input: [x_1, x_2],
                             var_output_expected: y
                             })
    loss_list.append(loss)
    
plt.scatter(range(100), loss_list)
plt.show()
    
    
  