import pymongo
import tensorflow as tf
import pandas as pd
import numpy as np
from util import ml

client = pymongo.MongoClient()
col = client.xingqiao.data
data = pd.DataFrame(list(col.find()))
data = [[i['birthDate'], i['times'], i['tengxunCreditScore'], i['status']] for i in col.find() if
        i.get('birthDate', 0) > 0 and i.get('tengxunCreditScore', 0) > 0]

batch_size = 1024
vec_size = len(data[0]) - 1

train_data = data[:-1 * batch_size]
train_data_neg = [i for i in train_data if i[-1] == 0]
cnt = sum([i[-1] for i in train_data]) - len(train_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(train_data))
    r2 = np.random.randint(len(train_data_neg))
    train_data.insert(r1, train_data_neg[r2 - 1])
train_data = pd.DataFrame(train_data)

test_data = data[-1 * batch_size:]
test_data_neg = [i for i in test_data if i[-1] == 0]
cnt = sum([i[-1] for i in test_data]) - len(test_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(test_data))
    r2 = np.random.randint(len(test_data_neg))
    test_data.insert(r1, test_data_neg[r2 - 1])
test_data = pd.DataFrame(test_data).sample(batch_size)


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

loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()

# ...................................................................
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 10 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_loss = sess.run(loss, feed_dict={x: x_test, y_: y_test})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        qtest = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size

        train_y = sess.run(y, feed_dict={x: x_train, y_: y_train})
        qtrain = 1 - len([i for i in train_y + y_train if 0.5 <= i <= 1.5]) / batch_size

        print(train_loss, test_loss, qtrain, qtest, sum(test_y * y_test) / sum(test_y),
              sum(test_y * y_test) / sum(y_test))
