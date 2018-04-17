import pymongo
import tensorflow as tf
import pandas as pd
import numpy as np
from util import ml

client = pymongo.MongoClient()
col = client.xingqiao.data
data = pd.DataFrame(list(col.find()))
print(data.columns)
data = data[['birthDate', 'times', 'tengxunCreditScore', 'status']]

data = data[data.birthDate > 0][data.tengxunCreditScore > 0]

data = np.array(data)

batch_size = 1024
vec_size = len(data[0]) - 1

r = []

while len(set(r)) < batch_size:
    r.append(np.random.randint(0, len(data)))

r = set(r)

train_data, test_data = [], []
for i in range(len(data)):
    if i not in r:
        train_data.append(data[i])
    else:
        test_data.append(data[i])

train_data, test_data = pd.DataFrame(data), pd.DataFrame(test_data)


def next():
    a = train_data.sample(batch_size)
    return np.array(a.drop(vec_size, axis=1)), np.array(a[vec_size])


x_test, y_test = np.array(test_data.drop(vec_size, axis=1)), np.array(test_data[vec_size])

print(sum(y_test) / len(y_test), '!!!!!!!!!!!!!!!!!!!!!!!!')

x = tf.placeholder(shape=[batch_size, vec_size], dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

lis = [ml.bn_with_wb(x)]
for i in range(20):
    lis.append(ml.res(lis[-1]))

lay1 = tf.nn.elu(ml.layer_basic(ml.bn_with_wb(lis[-1]), size=4))
y = tf.nn.sigmoid(ml.layer_basic(lay1, size=1)[:, 0])

#loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
loss=-tf.reduce_sum(tf.abs(y+y_-1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()

# ...................................................................
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 50 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_loss = sess.run(loss, feed_dict={x: x_test, y_: y_test})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        q = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size
        print(train_loss, test_loss, q)
