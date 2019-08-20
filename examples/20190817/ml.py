# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 10:52:24 2019

@author: johnc
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

data_x = np.random.normal(0.0, 1.0, [100])
data_y = data_x * 3 + 10

val_inp = tf.placeholder(tf.float32, shape=[1,])
val_exp = tf.placeholder(tf.float32, shape=[1,])
val_mult = tf.Variable(0.0, name='weight', dtype=tf.float32, trainable=True)
val_bias = tf.Variable(0.0, name='bias', dtype=tf.float32, trainable=True)

val_loss = tf.square(val_exp - (val_inp * val_mult + val_bias))
op_loss = tf.train.AdamOptimizer(0.01).minimize(val_loss)

smy_loss = tf.summary.scalar("Loss", tf.squeeze(val_loss))
smy_bias = tf.summary.scalar("Bias", val_bias)
smy_mult = tf.summary.scalar("Mult", val_mult)

smy_merged = tf.summary.merge_all()

epochs = 100

with tf.Session() as sess:
  smy_writer = tf.summary.FileWriter("summary/out_1", graph=sess.graph)
  sess.run(tf.global_variables_initializer())
  step = 0
  for epoch in range(epochs):
    for x, y in zip(data_x, data_y):
      _, smy = \
        sess.run([op_loss, smy_merged], feed_dict={
            val_inp: [x],
            val_exp: [y]
          })
      smy_writer.add_summary(smy, step)
      step += 1
      

    




