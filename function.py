import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from config import *


def read_train_data():
    data = pd.read_csv('train.csv')  # Read csv file in pandas dataframe
    labels = np.array(data.pop('label'))  # Remove the labels as a numpy array from the dataframe
    labels = LabelEncoder().fit_transform(labels)[:, None]
    labels = OneHotEncoder().fit_transform(labels).todense()
    data = StandardScaler().fit_transform(np.float32(data.values))  # Convert the dataframe to a numpy array
    data = data.reshape(-1, WIDTH, WIDTH, CHANNELS)  # Reshape the data into 42000 2d images
    train_data, valid_data = data[:-VALID_SIZE], data[-VALID_SIZE:]
    train_labels, valid_labels = labels[:-VALID_SIZE], labels[-VALID_SIZE:]
    return train_data, train_labels, valid_data, valid_labels


def init_params():
    w1 = tf.Variable(tf.truncated_normal([PATCH, PATCH, CHANNELS, DEPTH], stddev=0.1))
    b1 = tf.Variable(tf.zeros([DEPTH]))
    w2 = tf.Variable(tf.truncated_normal([PATCH, PATCH, DEPTH, 2 * DEPTH], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[2 * DEPTH]))
    w3 = tf.Variable(tf.truncated_normal([WIDTH // 4 * WIDTH // 4 * 2 * DEPTH, NEURONS], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape=[NEURONS]))
    w4 = tf.Variable(tf.truncated_normal([NEURONS, LABEL_CNT], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[LABEL_CNT]))
    return w1, b1, w2, b2, w3, b3, w4, b4


def logits(data, w1, b1, w2, b2, w3, b3, w4, b4):
    x = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b1)
    x = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b2)
    x = tf.reshape(x, (-1, WIDTH // 4 * WIDTH // 4 * 2*DEPTH))
    x = tf.nn.relu(tf.matmul(x, w3) + b3)
    return tf.matmul(x, w4) + b4