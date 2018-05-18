import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json

word_size = 100
embedding_size = 2
batch_size = 8192
# 长度大于10,正反各占一半,!里面有将近一般的短信重复,200以上
with open('/usr/local/oybb/project/xingqiao_data/msgTfidf556_train') as f:
    data = json.loads(f.read())
#..........................................................................
with open('/usr/local/oybb/project/xingqiao_data/msgTfidf556_test') as f:
    final_test_data = json.loads(f.read())
final_test_data_f = np.array([i for i in final_test_data if i[-1] == 0])
final_test_data_t = np.array([i for i in final_test_data if i[-1] == 1])

def test(batch_size=batch_size, times=10):
    score = []
    q=[]
    for i in range(times):
        r_t = np.random.randint(0, len(final_test_data_t), int(batch_size / 2))
        r_f = np.random.randint(0, len(final_test_data_f), int(batch_size / 2))
        data = np.concatenate([final_test_data_t[r_t], final_test_data_f[r_f]])
        out = sess.run((y_,y,loss), feed_dict={x: data[:, :-1], y_: data[:, -1]})
        score.append(out[2])
        q.append(1-len([i for i in out[0]+out[1]if 0.5<i<1.5])/batch_size)
    return np.mean(score),np.mean(q)
#..........................................................................
train_data_f = np.array([i for i in data if i[-1] == 0])
train_data_t = np.array([i for i in data if i[-1] == 1])[:len(train_data_f)]

data_for_test = np.concatenate([train_data_t[int(-1 * batch_size / 2):], train_data_f[int(-1 * batch_size / 2):]])
x_test, y_test = data_for_test[:,:-1],data_for_test[:,-1]

print('data pre-processing is done')

train_data_t=train_data_t[:int(-1*batch_size/2)]
train_data_f=train_data_f[:int(-1*batch_size/2)]
def next():
    r_t=np.random.randint(0,len(train_data_t),int(batch_size/2))
    r_f=np.random.randint(0,len(train_data_f),int(batch_size/2))
    data=np.concatenate([train_data_t[r_t],train_data_f[r_f]])
    return data[:,:-1],data[:,-1]


x = tf.placeholder(shape=[batch_size, word_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

embeddings = tf.Variable(
    tf.random_uniform([566+2, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)
X = tf.reshape(embed, [batch_size, word_size, embedding_size, 1])

c1 = ml.conv2d(X, conv_filter=[10, embedding_size, 1, 2], padding='VALID', ksize=[1, 10, 1, 1], pool_stride=[1, 4, 1, 1],
               pool_padding='SAME')
c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 5, 1, 1],
               pool_padding='SAME')
c3 = ml.conv2d(c2, conv_filter=[5, 1, 4, 8], padding='VALID', ksize=[1, 1, 1, 1], pool_stride=[1, 1, 1, 1],
               pool_padding='VALID')
# lay1 = tf.reshape(c2, [batch_size, -1])
# lay2 = ml.layer_basic(ml.bn(lay1), 1)
out = tf.reshape(c3, shape=[batch_size, 8])
y = tf.nn.sigmoid(ml.layer_basic(out, 1))[:, 0]
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# loss=tf.reduce_mean((y-y_)**2)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01, scope=None), weights_list=gv)
all_loss = loss + l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................', sum(y_test) / len(y_test))

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

        print(train_loss, test_loss, qtrain, qtest,test())

oo=list(pymongo.MongoClient().xingqiao.dataWithMsg.find())
w2i={i['word']:i['index'] for i in  pymongo.MongoClient().xingqiao.w2iGtTfidfGt5000.find()}
def text_test(text):
    jc=list(jieba.cut(text))
    l=([w2i.get(i,1)for i in jc]+[0]*100)[:100]
    l=[l]*batch_size
    return sess.run(y, feed_dict={x:l})[0]