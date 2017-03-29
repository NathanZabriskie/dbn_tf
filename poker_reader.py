import numpy as np
from os.path import join
from collections import namedtuple

NUM_CLASSES = 10
TEST = 'poker-hand-testing.data'
TRAIN = 'poker-hand-training-true.data'

FileData = namedtuple('FileData', 'training_data training_labels test_data test_labels')

def extract_labels(a):
    data = np.delete(a, -1, axis=1)
    labels = a[:, -1]

    num_instances = labels.shape[0]
    labels_out = np.zeros((num_instances, NUM_CLASSES))
    labels_out[np.arange(labels.shape[0]), labels] = 1.0

    return data, labels_out

def load_file(f_name):
    data = []
    with open(f_name, 'r') as f:
        lines = f.read()
        for line in lines.splitlines():
            nums = line.split(',')
            data.append([int(x) for x in nums])

    return np.array(data)

def load_set(directory):
    training = load_file(join(directory, TRAIN))
    training_data, training_labels = extract_labels(training)
    
    test = load_file(join(directory, TEST))
    test_data, test_labels = extract_labels(test)

    return FileData(
            training_data=training_data,
            training_labels=training_labels,
            test_data=test_data,
            test_labels=test_labels)

if __name__ == "__main__":
    load_set('data/poker')
