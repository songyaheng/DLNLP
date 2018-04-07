import math

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# TensorFlow集群描述信息，ps_hosts表示参数服务节点信息，worker_hosts表示worker节点信息
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# TensorFlow Server模型描述信息，包括作业名称，任务编号，隐含层神经元数量，MNIST数据目录以及每次训练数据大小（默认一个批次为100个图片）
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100, "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/notebooks/tmp", "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
FLAGS = tf.app.flags.FLAGS
#图片像素大小为28*28像素
IMAGE_PIXELS = 28
def main(_):
    #从命令行参数中读取TensorFlow集群描述信息
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # 创建TensorFlow集群描述对象
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 为本地执行Task，创建TensorFlow本地Server对象.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    #如果是参数服务，直接启动即可
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        #分配操作到指定的worker上执行，默认为该节点上的cpu0
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            # 定义TensorFlow隐含层参数变量，为全连接神经网络隐含层
            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS), name="hid_w")
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
            # 定义TensorFlow softmax回归层的参数变量
            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],  tddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
            #定义模型输入数据变量（x为图片像素数据，y_为手写数字分类）
            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            y_ = tf.placeholder(tf.float32, [None, 10])
            #定义隐含层及神经元计算模型
            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)
            #定义softmax回归模型，及损失方程
            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
            #定义全局步长，默认值为0
            global_step = tf.Variable(0)
            #定义训练模型，采用Adagrad梯度下降法
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            #定义模型精确度验证模型，统计模型精确度
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #对模型定期做checkpoint，通常用于模型回复
            saver = tf.train.Saver()
            #定义收集模型统计信息的操作
            summary_op = tf.merge_all_summaries()
            #定义操作初始化所有模型变量
            init_op = tf.initialize_all_variables()
            #创建一个监管程序，用于构建模型检查点以及计算模型统计信息。
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir="/tmp/train_logs", init_op=init_op, summary_op=summary_op, saver=saver, global_step=global_step, save_model_secs=600)
            #读入MNIST训练数据集
            mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
            #创建TensorFlow session对象，用于执行TensorFlow图计算
            with sv.managed_session(server.target) as sess:
                step = 0
                while not sv.should_stop() and step < 1000:
                    # 读入MNIST的训练数据，默认每批次为100个图片
                    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                    train_feed = {x: batch_xs, y_: batch_ys}
                    #执行分布式TensorFlow模型训练
                    _, step = sess.run([train_op, global_step], feed_dict=train_feed)
                    #每隔100步长，验证模型精度
                    if step % 100 == 0:
                        print("Done step %d" % step)
                        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            # 停止TensorFlow Session
            sv.stop()
if __name__ == "__main__":
    tf.app.run()