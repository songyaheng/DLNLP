#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf

b = tf.Variable(tf.truncated_normal([3,10, 1, 8],seed=2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(b))