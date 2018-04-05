from data.DataHelper import loadItemData2, loadSpuSearchAndTotal, batch_iter
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
total = loadSpuSearchAndTotal("/Users/songyaheng/Downloads/total.csv")
search = loadSpuSearchAndTotal("/Users/songyaheng/Downloads/search.csv")
m = {}
for k in search:
    sv = search.get(k)
    tv = total.get(k)
    v = 0
    if tv == 0:
        m[k] = v
    else:
        m[k] = sv / tv
x_data, y_label = loadItemData2("/Users/songyaheng/Downloads/data.xlsx", m)
# pca = PCA(n_components = 4)
# newData = np.log(pca.fit_transform(x_data))
min_max_scaler = preprocessing.MinMaxScaler()

X_scaled = min_max_scaler.fit_transform(x_data)

x = np.column_stack((X_scaled, y_label))
# data = pd.DataFrame(x)
data = pd.DataFrame(x, columns=["amount_180days",
                                "amount_30days",
                                "amount_7days",
                                "detail_uv_rate",
                                "gmv_180days",
                                "gmv_30days",
                                "gmv_7days",
                                "transfer_rate_30days",
                                "transfer_rate_7days", "label"])

# plt.hist(y_label)
# plt.show()
#
# print(data.corr())
print(data.shape)

plt.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号，如果是plt画图，则将mlp换成plt

sns.pairplot(data, x_vars=["amount_180days","amount_30days","amount_7days","detail_uv_rate",
                              "gmv_180days",
                              "gmv_30days",
                              "gmv_7days",
                              "transfer_rate_30days",
                              "transfer_rate_7days"], y_vars='label',kind="reg", size=9, aspect=0.2)
plt.show()
#
# min_max_scaler = preprocessing.MinMaxScaler()
#
# X_scaled = min_max_scaler.fit_transform(x_data)
#
# shuffle_indices = np.random.permutation(np.arange(len(y_label)))
# x_shuffled = X_scaled[shuffle_indices]
# y_shuffled = y_label[shuffle_indices]
#
# dev_sample_index = -1 * int(0.3 * float(len(y_label)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#
# # 参数定义
# learning_rate = 0.01
# training_epoch = 50
# batch_size = 1000
# display_step = 20
#
# x = tf.placeholder(tf.float32,[None, 9], name='X')
# y = tf.placeholder(tf.float32, [None, 1], name='Y')
#
# # 变量定义
# global_step = tf.Variable(0, name="global_step", trainable=False)
# W = tf.Variable(tf.random_normal([9, 1]), name="weight")
# b = tf.Variable(tf.ones([1]), name="bias")
#
# # 计算预测值
# pred = tf.matmul(x, W) + b
# loss = tf.reduce_mean(tf.square(y-pred))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     batchs = batch_iter(list(zip(x_train, y_train)), batch_size, training_epoch)
#     avg_cost = 0
#     for batch in batchs:
#         batch_xs, batch_ys = zip(*batch)
#         _, step, c = sess.run([optimizer, global_step, loss], feed_dict={x: batch_xs, y: batch_ys})
#         print("Epoch:", '%04d' % (step + 1), "cost=", "{:.9f}".format(c))
#         avg_cost = c / batch_size
#         plt.plot(step + 1, avg_cost, 'co')
#         if (step + 1) % display_step == 0:
#             acc = accuracy.eval({x: x_dev, y: y_dev})
#             print("Testing Accuracy:", acc)
#     print("Optimization Finished!")
#
#     w = sess.run(W)
#     b = list(sess.run(b))
#     print("权重：")
#     print(w)
#     print("偏值：")
#     print(b)
#     #归一化参数
#     print("归一化参数：")
#     print(min_max_scaler.data_max_)
#     print(min_max_scaler.data_min_)
#     print(min_max_scaler.data_range_)
#     plt.xlabel("Epoch")
#     plt.ylabel("Cost")
#     plt.show()
