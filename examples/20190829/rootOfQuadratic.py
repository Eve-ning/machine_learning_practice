# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:20:40 2019

@author: johnc
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# 3X^2 - 5X + 3 = y

data_x = np.random.normal(0, 0.2, [1000, 1])
data_y = 5 * data_x + 3

var_input = tf.placeholder(dtype=tf.float32, shape=[1])
var_w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[1,5]),
                     dtype=tf.float32)
var_b1 = tf.Variable(initial_value=tf.zeros([5]))
var_layer1 = \
  tf.nn.relu6(
    tf.matmul(tf.expand_dims(var_input, axis=0), var_w1) + var_b1)

var_w2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,1]),
                     dtype=tf.float32)
var_b2 = tf.Variable(initial_value=tf.zeros([1]))
var_output = tf.matmul(var_layer1, var_w2) + var_b2
var_expected = tf.placeholder(dtype=tf.float32, shape=[1])

var_loss = tf.sqrt(tf.square(var_expected - tf.squeeze(var_output)))
var_loss_op = tf.train.RMSPropOptimizer(0.0001).minimize(var_loss)

summary_loss = tf.summary.scalar("Loss", tf.squeeze(var_loss))
summary_w1 = tf.summary.histogram("w1", var_w1)
summary_b1 = tf.summary.histogram("b1", var_b1)
summary_w2 = tf.summary.histogram("w2", var_w2)
summary_b2 = tf.summary.histogram("b2", var_b2)
summary_merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("logs/hist")

epochs = 25

with tf.Session() as sess:
  step = 0
  sess.run(tf.global_variables_initializer())
  
  for epoch in range(epochs):
    for x, y in zip(data_x, data_y):
      
      _, loss, summary_str = \
        sess.run(
            [var_loss_op, var_loss, summary_merged],
            feed_dict = { var_input: x, var_expected: y })
      summary_writer.add_summary(summary_str, global_step=step)
      
      step += 1
      
      if (step % 100 == 0):
        print("Epoch \t {}, Step \t {}, Loss \t {}".format(
            epoch, step, loss
            ))
      
      
      
    



















