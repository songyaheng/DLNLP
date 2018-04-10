import tensorflow as tf
import time

# TensorFlow集群描述信息，ps_hosts表示参数服务节点信息，worker_hosts表示worker节点信息
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# TensorFlow Server模型描述信息，包括作业名称，任务编号，隐含层神经元数量，MNIST数据目录以及每次训练数据大小（默认一个批次为100个图片）
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# 假设输入数据已经用9.2.1小节中的方法转换成了单词编号的格式。
SRC_TRAIN_DATA = "./train.src"        # 源语言输入文件。
TRG_TRAIN_DATA = "./train.answer"        # 目标语言输入文件。
CHECKPOINT_PATH = "./seq2seq_ckpt"   # checkpoint保存路径。

HIDDEN_SIZE = 1024                   # LSTM的隐藏层规模。
NUM_LAYERS = 2                       # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000               # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000                # 目标语言词汇表大小。
BATCH_SIZE = 100                     # 训练数据batch的大小。
NUM_EPOCH = 5                        # 使用训练数据的轮数。
KEEP_PROB = 0.8                      # 节点不被dropout的概率。
MAX_GRAD_NORM = 5                    # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True         # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50   # 限定句子的最大单词数量。
SOS_ID  = 1    # 目标语言词汇表中<sos>的ID。

# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
# batching操作。
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据。
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
    # 由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空（只包含<EOS>）的句子和长度过长的句子。
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)

    # 从图9-5可知，解码器需要两种格式的目标句子：
    #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
    #   2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
    # 形式并加入到Dataset中。
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),      # 源句子是长度未知的向量
         tf.TensorShape([])),         # 源句子长度是单个数字
        (tf.TensorShape([None]),      # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),      # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))         # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset

# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    # 在forward函数中定义模型的前向计算图。
    # src_input, src_size, trg_input, trg_label, trg_size分别是上面
    # MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout。
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dynamic_rnn构造编码器。
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类
        # 张量的tuple，每个LSTMStateTuple对应编码器中的一层。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size,
        # max_time, HIDDEN_SIZE]。Seq2Seq模型中不需要用到enc_outputs，而
        # 后面介绍的attention模型会用到它。
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # 使用dyanmic_rnn构造解码器。
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是 [batch_size, max_time,
        # HIDDEN_SIZE]。
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        # 计算解码器每一步的log perplexity。这一步与语言模型代码相同。
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰
        # 模型的训练。
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))
        return cost_per_token, train_op

FLAGS = tf.app.flags.FLAGS
MOVING_AVERAGE_DECAY = 0.99

def main():

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
            # 定义初始化函数。
            initializer = tf.random_uniform_initializer(-0.05, 0.05)
            # 定义训练用的循环神经网络模型。
            with tf.variable_scope("nmt_model", reuse=None,
                           initializer=initializer):
                train_model = NMTModel()
            # 定义输入数据。
            data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
            iterator = data.make_initializable_iterator()

            (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
            # 定义前向计算图。输入数据以张量形式提供给forward函数。
            cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)
            #定义全局步长，默认值为0
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # 训练模型。
            saver = tf.train.Saver()

            is_chief = (FLAGS.task_index == 0)
            if is_chief:
                # 计算变量的滑动平均值。
                variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                with tf.control_dependencies([variables_averages_op, train_op]):
                    train_op = tf.no_op()
                print("Worker %d: Initializing session..." % FLAGS.task_index)
            else:
                print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

            hooks=[tf.train.StopAtStepHook(last_step=100000)]
            sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,
                                         device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                   checkpoint_dir="tmp/train_log",
                                                   hooks=hooks,
                                                   save_checkpoint_secs=60,
                                                   config=sess_config) as sess:
                print("Worker %d: Session initialization complete." % FLAGS.task_index)
                # Perform training
                time_begin = time.time()
                print("Training begins @ %f" % time_begin)
                local_step = 0
                step = 0
                while not sess.should_stop() and step < 100000:
                    _, step = sess.run(iterator.initializer, global_step)
                    cost, _ = sess.run([cost_op, train_op])
                    # 每200步保存一个checkpoint。
                    if step % 200 == 0:
                        print("After local-step %d global-step %d steps, per token cost is %.3f" % (local_step, step, cost))
                        saver.save(sess, CHECKPOINT_PATH, global_step=step)
                    local_step += 1
                    now = time.time()
                    print("%f: Worker %d: training step %d done (global step: %d)" %
                          (now, FLAGS.task_index, local_step, step))
                saver.save(sess, CHECKPOINT_PATH, global_step=step)
if __name__ == "__main__":
    main()