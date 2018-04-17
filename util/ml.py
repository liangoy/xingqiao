import tensorflow as tf


def bn(x):
    mean, var = tf.nn.moments(x, axes=[0])
    var += 0.1 ** 7
    hat = (x - mean) / tf.sqrt(var)
    return hat


def bn_with_wb(x):
    w = tf.Variable(tf.random_uniform([x.shape[1].value], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([x.shape[1].value], -1.0, 1.0))
    return bn(x) * w + b


def layer_basic(x, size=0, with_b=True):
    if not size:
        size = x.shape[1].value
    w = tf.Variable(tf.random_uniform([x.shape[1].value, size], -1.0, 1.0))
    if with_b:
        b = tf.Variable(tf.random_uniform([size], -1.0, 1.0))
        return tf.matmul(x, w) + b
    else:
        return tf.matmul(x, w)


def res(x):
    lay1 = tf.nn.elu(layer_basic(bn_with_wb(x)))
    lay2 = tf.nn.elu(layer_basic(bn_with_wb(lay1)))
    lay3 = tf.nn.elu(layer_basic(bn_with_wb(lay2)))
    lay4 = tf.nn.elu(layer_basic(bn_with_wb(lay3)))
    lay5 = tf.nn.elu(layer_basic(bn_with_wb(lay4)))
    return lay5 + x
