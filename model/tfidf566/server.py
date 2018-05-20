import tensorflow as tf
import jieba
import pymongo
import numpy as np
from util import ml
import json
import jieba


class Server():
    db = pymongo.MongoClient().xingqiao
    w2i = {i['word']: i['index'] for i in db.w2iGtTfidfGt5000.find()}

    def __init__(self):
        word_size = 100
        embedding_size = 2
        self.batch_size = 8192
        # ...........................................................................
        with open('/usr/local/oybb/project/xingqiao_data/msgTfidf566') as f:
            data = json.loads(f.read())
        data = [(i + [0] * word_size)[:word_size + 2] for i in data]

        data_f = np.array([i for i in data if i[1] == 0 and i[0] == -1])
        data_t = np.array([i for i in data if i[1] == -1 and i[0] == -1])[:len(data_f)]
        self.data = np.concatenate([data_f, data_t])[:, 2:]
        # ..........................................................................
        print('data pre-processing is done')

        self.x = tf.placeholder(shape=[self.batch_size, word_size], dtype=tf.int32)
        self.y_ = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

        embeddings = tf.Variable(
            tf.random_uniform([566 + 2, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.x)
        X = tf.reshape(embed, [self.batch_size, word_size, embedding_size, 1])

        c1 = ml.conv2d(X, conv_filter=[10, embedding_size, 1, 2], padding='VALID', ksize=[1, 10, 1, 1],
                       pool_stride=[1, 4, 1, 1],
                       pool_padding='SAME')
        c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 5, 1, 1],
                       pool_padding='SAME')
        c3 = ml.conv2d(c2, conv_filter=[5, 1, 4, 8], padding='VALID', ksize=[1, 1, 1, 1], pool_stride=[1, 1, 1, 1],
                       pool_padding='VALID')

        out = ml.bn_with_wb(tf.reshape(c3, shape=[self.batch_size, 8]))
        self.y = tf.nn.sigmoid(ml.layer_basic(out, 1))[:, 0]
        # ...................................................................
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, '/usr/local/oybb/project/xingqiao_model/./tfidf566')

    def cut_word(self, text):
        return ([self.w2i.get(i, 1) for i in jieba.cut(text)] + [0] * 100)[:100]

    def get_score(self, text_list, m='r'):
        score = []
        text_list_cut1 = [self.cut_word(i) for i in text_list[:1000]]
        text_list_cut2 = [self.cut_word(i) for i in text_list[1000:2000]]
        if text_list_cut1:
            r1 = np.random.randint(0,len(self.data),self.batch_size - len(text_list_cut1))
            data = np.concatenate([self.data[r1], text_list_cut1])
            score.extend(list(self.sess.run(self.y, feed_dict={self.x: data})[len(r1):]))
        if text_list_cut2:
            r1 = np.random.randint(0,len(self.data),self.batch_size - len(text_list_cut2))
            data = np.concatenate([self.data[r1], text_list_cut2])
            score.extend(list(self.sess.run(self.y, feed_dict={self.x: data})[len(r1):]))
        score = [1 if i>0.5 else 0 for i in score]
        return (0.52/0.5)**sum(score)*(0.48/0.5)**(len(score)-sum(score))
