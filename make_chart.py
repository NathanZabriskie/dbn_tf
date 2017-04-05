import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt

def make_acc_filename(p):
    return os.path.join(p, 'accuracy.pickle')

def make_loss_filename(p):
    return os.path.join(p, 'loss.pickle')

def load_folder(p):
    with open(make_acc_filename(p), 'rb') as f:
        a = pickle.load(f)

    with open(make_loss_filename(p), 'rb') as f:
        l = pickle.load(f)

    return a, l

def make_figure(l1, l2, y_axis_name, min_y, max_y):
    plt.figure(1, figsize=(8,4))

    plt.subplot(121)
    plt.plot(l1)
    plt.xlabel('epochs')
    plt.ylabel(y_axis_name)
    plt.ylim(ymin=min_y, ymax=max_y)

    plt.subplot(122)
    plt.plot(l2)
    plt.xlabel('epochs')
    plt.ylabel(y_axis_name)
    plt.ylim(ymin=min_y, ymax=max_y)

    plt.show()
    plt.close()

def make_single_figure(l1, y_axis_name, min_y, max_y):
    plt.figure(1, figsize=(5,4))
    plt.plot(l1)
    plt.xlabel('epochs')
    plt.ylabel(y_axis_name)
    plt.ylim(ymin=min_y, ymax=max_y)
    plt.show()
    plt.close()

if __name__ == '__main__':
    a1, l1 = load_folder(sys.argv[1])
    a2, l2 = load_folder(sys.argv[2])

    #make_figure(a1, a2, 'validation set accuracy', 0.5, 1.0)
    make_single_figure(a1, 'validation set accuracy', 0.9, 1.0)
    make_single_figure(a2, 'validation set accuracy', 0.9, 1.0)
