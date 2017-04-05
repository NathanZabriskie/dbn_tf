import math
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
HIDDEN1_UNITS = 200
OUTPUT_UNITS = 10
NUM_TRAINING = 60000

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=1000)
training = mnist.train.images[0:NUM_TRAINING,:]
training_labels = mnist.train.labels[0:NUM_TRAINING,:]

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    LABELS = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('reshape'):
    XX = tf.reshape(X, [-1, IMAGE_PIXELS])

with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, HIDDEN1_UNITS],
                    stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
    biases = tf.Variable(tf.zeros([HIDDEN1_UNITS]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(XX, weights) + biases)

with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([HIDDEN1_UNITS, OUTPUT_UNITS],
                    stddev=1.0/math.sqrt(float(HIDDEN1_UNITS))),
                name='weights')
    biases = tf.Variable(tf.zeros([OUTPUT_UNITS]), name='biases')
    Ylogits = tf.matmul(hidden1, weights) + biases
    # Sigmoid output
    #Y = tf.nn.sigmoid(Ylogits)
    # Softmax output
    Y = tf.nn.softmax(Ylogits)

#loss = tf.sqrt(tf.reduce_mean(tf.square(Y - LABELS)))

with tf.name_scope('loss_function'):
    loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=Ylogits,
                labels=LABELS, name='xentropy'
            )
    loss = tf.reduce_mean(loss, name='xentropy_mean')

tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(LABELS, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)



global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

epochs = 0
since_improved = 0
best_loss = 99999.0
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    validation_writer = tf.summary.FileWriter('logs/train', sess.graph)
    start = 0
    while(since_improved < 5):
        if start + 100 > training.shape[0]:
            start = 0
            unison_shuffle(training, training_labels)
            epochs += 1
            loss_, summ = sess.run([loss, merged], feed_dict={X: mnist.validation.images, LABELS: mnist.validation.labels})
            validation_writer.add_summary(summ, epochs)
            if loss_ < best_loss:
                print('Best validation loss=', loss_)
                since_improved = 0
                best_loss = loss_
                saver.save(sess, 'saves/test.ckpt', global_step=step)
            else:
                since_improved += 1

        images, labels = training[start:start+100], training_labels[start:start+100]
        start += 100
        _, loss_, step = sess.run([train_op, loss, global_step], feed_dict={X: images, LABELS: labels})

    validation_writer.close()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, tf.train.latest_checkpoint('saves/')) # restore the best model as measured on validation set.
    loss_, acc = sess.run([loss, accuracy], feed_dict={X: mnist.test.images, LABELS: mnist.test.labels})
    print('Final Test\nAcc=', acc, '\nLoss=', loss_)
