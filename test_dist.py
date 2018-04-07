import tensorflow as tf
import numpy as np
import time

tf.app.flags.DEFINE_string("server", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

with tf.Session(FLAGS.server) as session:
    train_X = np.linspace(-1.0, 1.0, 100)
    train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    loss = tf.square(Y - X * w - b)

    global_step = tf.Variable(0)

    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        tf.global_variables_initializer().run()

        for (x, y) in zip(train_X, train_Y):
            _, step = session.run([train_op, global_step],
                           feed_dict={X: x,Y: y})

            loss_value = session.run(loss, feed_dict={X: x, Y: y})
            print("Step: {}, loss: {}".format(step, loss_value))
            time.sleep(3)