import sys
import tensorflow as tf

try:
    worker1 = "10.244.2.140:2222"
    worker2 = "10.244.1.134:2222"
    worker3 = "10.244.2.141:2222"
    worker_hosts = [worker1, worker2, worker3]
    cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})
    server = tf.train.Server(cluster_spec, job_name="worker", task_index=2)
    server.join()
except KeyboardInterrupt:
    sys.exit()