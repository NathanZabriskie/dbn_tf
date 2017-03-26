from model.dbn import DBN
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=6000)

deep = DBN(hidden_layers=[500,450,500])
toy_test = np.random.uniform(size=[10,5,5,1])

flattened_train = np.reshape(mnist.train.images, [mnist.train.images.shape[0], -1])
flattened_validation = np.reshape(mnist.validation.images,
                                  [mnist.validation.images.shape[0], -1])
flattened_test = np.reshape(mnist.test.images, [mnist.test.images.shape[0], -1])
deep.pretrain(flattened_train)
deep.train(train_set=flattened_train,
           train_labels=mnist.train.labels,
           validation_set=flattened_validation,
           validation_labels=mnist.validation.labels)
