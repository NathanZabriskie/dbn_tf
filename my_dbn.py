from model.dbn import DBN
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy.io.arff import loadarff

# Session settings
SAVE_DIR = 'saves/t1'
OUTPUT_DIR = 'results/mnist_pretrain_momentum_dropout_JUMBO_smallLR2_win'
if os.path.exists(OUTPUT_DIR):
    choice = input(OUTPUT_DIR + ' already exists. Do you want to overwrite these results? y/n')
    if choice != 'y':
        print('Exiting')
        exit()

PRETRAIN_ITERATIONS = 1
LEARNING_RATE = 0.001
DECAY_LR = False
FREEZE_RBMS = False
RBM_ACTIVATION = 'sigmoid'
RBM_LAYERS = [600,625,650,600]
KEEP_CHANCE = 0.8

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=6000)

deep = DBN(hidden_layers=RBM_LAYERS,
           rbm_activation=RBM_ACTIVATION,
           freeze_rbms=FREEZE_RBMS,
           keep_chance=KEEP_CHANCE)

flattened_train = np.reshape(mnist.train.images, [mnist.train.images.shape[0], -1])
flattened_validation = np.reshape(mnist.validation.images,
                                  [mnist.validation.images.shape[0], -1])
flattened_test = np.reshape(mnist.test.images, [mnist.test.images.shape[0], -1])

deep.pretrain(train_set=flattened_train,
              pretrain_iterations=PRETRAIN_ITERATIONS,
              learning_rate=LEARNING_RATE)

loss_hist, acc_hist = deep.train(train_set=flattened_train,
                                 train_labels=mnist.train.labels,
                                 validation_set=flattened_validation,
                                 validation_labels=mnist.validation.labels,
                                 save_dir=SAVE_DIR,
                                 learning_rate=LEARNING_RATE,
                                 decay_lr=DECAY_LR)

accuracy, loss = deep.measure_test_accuracy(test_set=flattened_test,
                                            test_labels=mnist.test.labels,
                                            save_dir=SAVE_DIR)

print('Final Accuracy:', accuracy)
print('Final Loss:', loss)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, 'loss.pickle'), 'wb') as f:
    pickle.dump(loss_hist, f)

with open(os.path.join(OUTPUT_DIR, 'accuracy.pickle'), 'wb') as f:
    pickle.dump(acc_hist, f)

with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write('RESULTS:\n')
    f.write('Trained in ' + str(len(loss_hist)) + ' epochs.\n')
    f.write('LR = ' + str(LEARNING_RATE) + '\n')
    f.write('Training dropout keep chance = ' + str(KEEP_CHANCE) + '\n')
    f.write('DECAY_LR = ' + str(DECAY_LR) + '\n')
    f.write('PRETRAIN_ITERATIONS = ' + str(PRETRAIN_ITERATIONS) + '\n')
    f.write('RBM_LAYERS = ' + str(RBM_LAYERS) + '\n')
    f.write('Freeze RBMS = ' + str(FREEZE_RBMS) + '\n')
    f.write('RBM activation = ' + str(RBM_ACTIVATION) + '\n')
    f.write('Final Accuracy = ' + str(accuracy) + '\n')
    f.write('Final Cross Entropy Loss = ' + str(loss) + '\n')


plt.plot(loss_hist)
plt.ylabel('Mean Cross Entropy Loss')
plt.xlabel('Epochs')
plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'))
plt.close()

plt.plot(acc_hist)
plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')
plt.savefig(os.path.join(OUTPUT_DIR, 'acc.png'))
plt.close()
