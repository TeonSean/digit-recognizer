from sklearn.model_selection import ShuffleSplit

from function import *
from config import *

tf_data = tf.placeholder(tf.float32, shape=(None, WIDTH, WIDTH, CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABEL_CNT))
w1, b1, w2, b2, w3, b3, w4, b4 = init_params()
tf_pred = tf.nn.softmax(logits(tf_data, w1, b1, w2, b2, w3, b3, w4, b4))
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits(tf_data, w1, b1, w2, b2, w3, b3, w4, b4),
                                                                 labels=tf_labels))
tf_acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_labels, 1))))
tf_opt = tf.train.RMSPropOptimizer(LEARNING_RATE)
tf_step = tf_opt.minimize(tf_loss)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
ss = ShuffleSplit(n_splits=STEP, train_size=BATCH)
train_data, train_labels, valid_data, valid_labels = read_train_data()
ss.get_n_splits(train_data, train_labels)
for step, (idx, _) in enumerate(ss.split(train_data,train_labels), start=1):
    fd = {tf_data: train_data[idx], tf_labels: train_labels[idx]}
    session.run(tf_step, feed_dict=fd)
    if step % 500 == 0:
        fd = {tf_data:valid_data, tf_labels:valid_labels}
        valid_loss, valid_accuracy = session.run([tf_loss, tf_acc], feed_dict=fd)
        print('Step %i \t Valid. Acc. = %f' % (step, valid_accuracy), end='\n')
test = pd.read_csv('test.csv')
test_data = StandardScaler().fit_transform(np.float32(test.values))
test_data = test_data.reshape(-1, WIDTH, WIDTH, CHANNELS)
test_pred = session.run(tf_pred, feed_dict={tf_data: test_data})
test_labels = np.argmax(test_pred, axis=1)
submission = pd.DataFrame(data={'ImageId': (np.arange(test_labels.shape[0])+1), 'Label': test_labels})
submission.to_csv('submission.csv', index=False)
submission.tail()
