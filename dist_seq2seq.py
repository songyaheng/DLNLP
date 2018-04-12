import argparse
import sys,os
import tensorflow as tf
import time
import numpy as np
import collections
import math

FLAGS = None
#图片像素大小为28*28像素
IMAGE_PIXELS = 28
class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(train_dir,
                   reshape=True,
                   validation_size=2000):
    trainfile = os.path.join(train_dir, "mnist_train.csv")
    testfile = os.path.join(train_dir, "mnist_test.csv")
    train_images = np.array([], dtype=np.uint8)
    train_labels = np.array([], dtype=np.uint8)
    test_images = np.array([], dtype=np.uint8)
    test_labels = np.array([], dtype=np.uint8)

    count = 0
    with open(trainfile) as f:
        for line in f.readlines():
            count += 1
            line = line.strip()
            line = line.split(",")
            line = [int(x) for x in line]
            one_rray = np.array(line[1:], dtype=np.uint8)
            train_images = np.hstack((train_images, one_rray))
            train_labels = np.hstack((train_labels, np.array(line[0], dtype=np.uint8)))
            if count % 10000 == 0:
                print(str(count))
            if count == 20000:
                break
    train_images = train_images.reshape(20000, 28*28)
    train_labels = train_labels.reshape(20000, 1)
    train_labels = dense_to_one_hot(train_labels, 10)

    count = 0
    with open(testfile) as f:
        for line in f.readlines():
            count += 1
            line = line.strip()
            line = line.split(",")
            line = [int(x) for x in line]
            one_rray = np.array(line[1:], dtype=np.uint8)
            test_images = np.hstack((test_images, one_rray))
            test_labels = np.hstack((test_labels, np.array(line[0], dtype=np.uint8)))
            if count % 10000 == 0:
                print(str(count))
    test_images = test_images.reshape(10000, 28*28)
    test_labels = test_labels.reshape(10000, 1)
    test_labels = dense_to_one_hot(test_labels, 10)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, reshape=reshape)
    test = DataSet(test_images, test_labels, reshape=reshape)


    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return Datasets(train=train, validation=validation, test=test)

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
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # 定义TensorFlow隐含层参数变量，为全连接神经网络隐含层
            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS), name="hid_w")
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
            # 定义TensorFlow softmax回归层的参数变量
            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="sm_w")
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
            global_step = tf.train.get_or_create_global_step()
            #定义训练模型，采用Adagrad梯度下降法
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            #定义模型精确度验证模型，统计模型精确度
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #对模型定期做checkpoint，通常用于模型回复
        #读入MNIST训练数据集
        mnist = read_data_sets(FLAGS.data_dir)
        saver = tf.train.Saver()
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.train_epoch)]
        # 通过tf.train.MonitoredTrainingSession管理训练深度学习模型的通用功能。
        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=(FLAGS.task_index == 0),
                                            checkpoint_dir=FLAGS.data_dir,
                                            hooks=hooks,
                                            save_checkpoint_secs=10,
                                            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            print("Worker %d: Session initialization complete." % FLAGS.task_index)
            local_step = 0
            step = 0
            while not sess.should_stop():
                # 读入MNIST的训练数据，默认每批次为100个图片
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_: batch_ys}
                #执行分布式TensorFlow模型训练
                _, step = sess.run([train_op, global_step], feed_dict=train_feed)
                local_step = local_step + 1
                now = time.time()
                print("%f: Worker %d: training step %d done (global step: %d)" %
                        (now, FLAGS.task_index, local_step, step))
                #每隔100步长，验证模型精度
                if step % 100 == 0:
                    print("acc: %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
                    print("cross entropy = %g" % sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("acc: %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("cross entropy = %g" % sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            saver.save(sess, FLAGS.data_dir, global_step=step)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tfdata/tmp",
        help="Index of task within the job"
    )
    parser.add_argument(
        "--hidden_units",
        type=float,
        default=100,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--train_epoch",
        type=int,
        default=100000,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)