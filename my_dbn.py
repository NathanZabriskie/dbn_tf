from model.dbn import DBN
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import os
# Session settings
SAVE_DIR = 'saves/t1'
OUTPUT_DIR = 'results/mnist1'

PRETRAIN_ITERATIONS = 10000
LEARNING_RATE = 0.01
DECAY_LR = False
FREEZE_RBMS = True

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=6000)

deep = DBN(hidden_layers=[500,450,500])
toy_test = np.random.uniform(size=[10,5,5,1])

flattened_train = np.reshape(mnist.train.images, [mnist.train.images.shape[0], -1])
flattened_validation = np.reshape(mnist.validation.images,
                                  [mnist.validation.images.shape[0], -1])
flattened_test = np.reshape(mnist.test.images, [mnist.test.images.shape[0], -1])

deep.pretrain(train_set=flattened_train,
              pretrain_iterations=PRETRAIN_ITERATIONS,
              learning_rate=LEARNING_RATE)

loss_hist = deep.train(train_set=flattened_train,
                       train_labels=mnist.train.labels,
                       validation_set=flattened_validation,
                       validation_labels=mnist.validation.labels,
                       save_dir=SAVE_DIR,
                       learning_rate=LEARNING_RATE,
                       decay_lr=DECAY_LR,
                       freeze_rbms=FREEZE_RBMS)

accuracy, loss = deep.measure_test_accuracy(test_set=flattened_test,
                                            test_labels=mnist.test.labels,
                                            save_dir=SAVE_DIR)

print('Final Accuracy:', accuracy)
print('Final Loss:', loss)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write('RESULTS:\n')
    f.write('Trained in ' + str(len(loss_hist)) + ' epochs.\n')
    f.write('LR = ' + str(LEARNING_RATE) + '\n')
    f.write('DECAY_LR = ' + str(DECAY_LR) + '\n')
    f.write('Freeze RBMS = ' + str(FREEZE_RBMS) + '\n')
    f.write('Final Accuracy = ' + str(accuracy) + '\n')
    f.write('Final Cross Entropy Loss = ' + str(loss) + '\n')
