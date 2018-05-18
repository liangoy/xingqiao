import tensorflow as tf
import json
import numpy as np
from util import ml

text_size = 1000 * 2
embedding_size = 1
batch_size =512
with open('/usr/local/oybb/project/xingqiao_data/msgTfidf') as f:
    data = json.loads(f.read())

data = [(i + [0] * text_size)[:text_size + 1] for i in data]

data_t = np.array([i for i in data if i[0] == 1])
data_f = np.array([i for i in data if i[0] == 0])

train_data_t = data_t[:int(-1 * batch_size / 2)]
train_data_f = data_f[:int(-1 * batch_size / 2)]

test_data=np.concatenate([data_t[int(-1*batch_size/2):],data_f[int(-1*batch_size/2):]])

def next():
    r_t = np.random.randint(0, len(train_data_t), int(batch_size / 2))
    r_f = np.random.randint(0, len(train_data_f), int(batch_size / 2))
    data = np.concatenate([train_data_t[r_t], train_data_f[r_f]])
    return data[:, 1:], data[:, 0]


x_test, y_test = test_data[:,1:], test_data[:, 0]
x = tf.placeholder(shape=[batch_size, text_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

embeddings = tf.Variable(
    tf.random_uniform([566 + 3, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x + 1)
X = tf.reshape(embed, [batch_size, text_size, embedding_size, 1])

c1 = ml.conv2d(X, conv_filter=[3, embedding_size, 1, 2], padding='VALID', ksize=[1, 20, 1, 1],
               pool_stride=[1, 20, 1, 1],
               pool_padding='SAME')
c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 10, 1, 1],
               pool_padding='SAME')
c3 = ml.conv2d(c2, conv_filter=[2, 1, 4, 8], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 10, 1, 1],
               pool_padding='VALID')

out = ml.bn(tf.reshape(c3, shape=[batch_size, 8]))
y = tf.nn.sigmoid(ml.layer_basic(out,1))[:, 0]
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1.0 - y_) * tf.log(1.0 - y + 0.00000001)) / batch_size / tf.log(
    2.0)
gv= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#loss=tf.reduce_mean((y-y_)**2)
l2_loss=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.05, scope=None), weights_list=gv)
all_loss=loss+l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................', sum(y_test) / len(y_test))

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 20 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_loss = sess.run(loss, feed_dict={x: x_test, y_: y_test})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        qtest = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size

        train_y = sess.run(y, feed_dict={x: x_train, y_: y_train})
        qtrain = 1 - len([i for i in train_y + y_train if 0.5 <= i <= 1.5]) / batch_size

        print(train_loss, test_loss, qtrain, qtest)
