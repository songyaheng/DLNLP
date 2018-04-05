import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from data.DataHelper import loadItemData, batch_iter

x_data, y_label = loadItemData("/Users/songyaheng/Downloads/data.xlsx")

min_max_scaler = preprocessing.MinMaxScaler()

X_scaled = min_max_scaler.fit_transform(x_data)

shuffle_indices = np.random.permutation(np.arange(len(y_label)))
x_shuffled = X_scaled[shuffle_indices]
y_shuffled = y_label[shuffle_indices]

dev_sample_index = -1 * int(0.3 * float(len(y_label)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# 参数定义
learning_rate = 0.01
training_epoch = 50
batch_size = 1000
display_step = 20

x = tf.placeholder(tf.float32,[None, 9], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='Y')

# 变量定义
global_step = tf.Variable(0, name="global_step", trainable=False)
W = tf.Variable(tf.zeros([9, 1]), name="weight")
b = tf.Variable(tf.ones([1]), name="bias")

# 计算预测值
pred = tf.matmul(x, W) + b
predict = tf.nn.sigmoid(pred)
# 计算损失值 使用相对熵计算损失值
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    batchs = batch_iter(list(zip(x_train, y_train)), batch_size, training_epoch)
    avg_cost = 0
    for batch in batchs:
        batch_xs, batch_ys = zip(*batch)
        _, step, c = sess.run([optimizer, global_step, loss], feed_dict={x: batch_xs, y: batch_ys})
        print("Epoch:", '%04d' % (step + 1), "cost=", "{:.9f}".format(c))
        avg_cost = c / batch_size
        plt.plot(step + 1, avg_cost, 'co')
        if (step + 1) % display_step == 0:
            acc = accuracy.eval({x: x_dev, y: y_dev})
            print("Testing Accuracy:", acc)
    print("Optimization Finished!")

    w = sess.run(W)
    b = list(sess.run(b))
    print("权重：")
    print(w)
    print("偏值：")
    print(b)
    #归一化参数
    print("归一化参数：")
    print(min_max_scaler.data_max_)
    print(min_max_scaler.data_min_)
    print(min_max_scaler.data_range_)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
