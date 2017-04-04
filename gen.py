from model.dbn import DBN
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import sys
import os
import matplotlib.pyplot as plt


SAVE_DIR = 'saves/gennn'
OUTPUT_DIR = 'results/mnist_generated_10000'

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

PRETRAIN_ITERATIONS = 10000
LEARNING_RATE = 0.001
DECAY_LR = False
FREEZE_RBMS = False
RBM_ACTIVATION = 'sigmoid'
RBM_LAYERS = [600,625,IMAGE_PIXELS]
KEEP_CHANCE = 0.8

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=6000)

deep = DBN(hidden_layers=RBM_LAYERS,
           rbm_activation=RBM_ACTIVATION,
           freeze_rbms=FREEZE_RBMS,
           keep_chance=KEEP_CHANCE)

flattened_train = np.reshape(mnist.train.images, [mnist.train.images.shape[0], -1])

deep.pretrain(train_set=flattened_train,
              pretrain_iterations=PRETRAIN_ITERATIONS,
              learning_rate=LEARNING_RATE)

deep.build_graph(flattened_train.shape[1], mnist.train.labels.shape[1])
deep.build_gen_graph(True)

imgs, labels = deep.generate_example(train_set=flattened_train, train_labels=mnist.train.labels, out_dir=OUTPUT_DIR, save_dir=SAVE_DIR)
