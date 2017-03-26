import numpy as np
from yadlt.models.boltzmann import dbn
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=6000)

trainImages = np.reshape(mnist.train.images, [-1, IMAGE_PIXELS])
print(trainImages.shape)
testImages = np.reshape(mnist.test.images, [-1, IMAGE_PIXELS])
validationImages = np.reshape(mnist.validation.images, [-1, IMAGE_PIXELS])

deepBelief = dbn.DeepBeliefNetwork(rbm_layers=[400,500,450])
deepBelief.pretrain(trainImages, validationImages)
deepBelief.build_model(n_features=IMAGE_PIXELS, n_classes=10)
deepBelief.fit(train_X=trainImages, train_Y=mnist.train.labels,
               val_X=validationImages, val_Y=mnist.validation.labels)
