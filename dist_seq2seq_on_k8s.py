import tensorflow as tf
import time
import argparse
import sys
FLAGS = None
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
            while not sess.should_stop():

                local_step = local_step + 1
                now = time.time()
                print("%f: Worker %d: training step %d done (global step: %d)" %
                      (now, FLAGS.task_index, local_step, step))
                #每隔100步长，验证模型精度
                if step % 100 == 0:
                    cost, _ = sess.run([cost_op, train_op])
                    print("cross entropy = %g" % cost)
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
        default="worker",
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
        default="/tfdata/seq2seq",
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
