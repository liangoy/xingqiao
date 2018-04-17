import tensorflow as tf
import pandas as pd
import numpy as np
from util import ml

train_data = pd.read_excel('/home/liangoy/Downloads/p2ptrain.xlsx')
test_data = pd.read_excel('/home/liangoy/Downloads/p2ptest.xlsx')

train_data = pd.concat((train_data, test_data))

batch_size = len(test_data)


def next():
    a = train_data.sample(batch_size)
    return np.array(a.drop('公司状态', axis=1)), np.array(a['公司状态'])


x_test, y_test = np.array(test_data.drop('公司状态', axis=1)), np.array(test_data['公司状态'])

x = tf.placeholder(shape=[batch_size, 12], dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

lay1 = ml.layer_basic(ml.bn(x), size=1)[:, 0]

y = tf.nn.sigmoid(lay1)

loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()

# ...................................................................
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 100 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        q = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size
        print(train_loss, q)
