import numpy as np
import tensorflow as tf

def make_full_layer(graph, input_tensor, num_units, activation):


class RBMLayer:
    def __init__(self, num_hidden, name):
        self.W = None
        self.bias_visible = None
        self.bias_hidden = np.zeros([1, num_hidden])

        self.num_hidden = num_hidden
        self.name = name

    def activate_hidden_from_visible(self, visible):
        net = np.dot(visible, self.W)
        return self.activate(net + self.bias_hidden)

    def sample_hidden_from_visible(self, visible):
        return self.sample(self.activate_hidden_from_visible(visible))

    def activate_visible_from_hidden(self, hidden):
        net = np.dot(hidden, self.W.T)
        return self.activate(net + self.bias_visible)

    def sample_visible_from_hidden(self, hidden):
        return self.sample(self.activate_visible_from_hidden(hidden))

    def activate(self, nets):
        return 1.0 / (1.0 + np.exp(-1.0 * nets))

    def sample(self, probabilities):
        return np.floor(probabilities + np.random.uniform(size=probabilities.shape))

    def RBMUpdate(self, x1, learning_rate):
        if self.W is None or self.bias_visible is None:
            self.W = np.random.normal(scale=0.1, size=[x1.shape[1], self.num_hidden])
            self.bias_visible = np.zeros([1, x1.shape[1]])

        h1 = self.sample_hidden_from_visible(x1)

        x2 = self.sample_visible_from_hidden(h1)
        h_prob = self.activate_hidden_from_visible(x2)

        delta_W = learning_rate * (np.dot(h1.T, x1))
        self.bias_hidden += learning_rate * (h1 - h_prob)
        self.bias_visible += learning_rate * (x1 - x2)
        self.W += delta_W.T
        return np.sum(np.absolute(delta_W))

    def build_graph(self, graph, input_tensor, is_frozen=True, is_sampled=False):
        with graph.as_default():
            with tf.name.scope(self.name):
                if is_frozen:
                    weights = tf.constant(self.W, name='weights')
                else:
                    weights = tf.Variable(self.W, name='weights')

                bias = tf.constant(self.bias_hidden, name='bias')
                if is_sampled:
                    return
                else:
                    return tf.nn.sigmoid(tf.matmul(input_tensor, weights) + bias)

class DBN:
    def __init__(self, hidden_layers, fully_connected_layers=[200],
                 output_activation='softmax', name='dbn'):
        self.rbms = []
        for l in hidden_layers:
            name = 'rbm_' + str(l)
            self.rbms.append(RBMLayer(l, name))

        self.fully_connected_layers = []

        for l in fully_connected_layers:
            name = 'fullLayer_' + str(l)
            self.fully_connected_layers.append(FullLayer(l, name))

        self.output_activation = output_activation
        self.name = name

    def pretrain(self, train_set, learning_rate=0.01):
        train = train_set
        if len(train.shape) != 2:
            train = self.flatten_array(train)


        for i, layer in enumerate(self.rbms):
            best_loss = 999999.0
            since_improve = 0
            lr = learning_rate
            print('Pretraining layer', i, 'of', len(self.rbms))
            for j in range(10000):
                if j % 1000 == 0:
                    print('.')
                output = train[np.random.randint(0,train.shape[0],1)]
                for rbm in self.rbms[:i]:
                    output = rbm.sample_hidden_from_visible(output)

                loss = layer.RBMUpdate(output, lr)
                if loss < best_loss:
                    best_loss = loss
                    since_improve = 0
                elif since_improve > 1000:
                    lr *= 0.9
                    since_improve = 0
                else:
                    since_improve += 1

        print('Finished Pre-Training')

    def flatten_array(self, np_array):
        size = 1
        for dim in np_array.shape[1:]:
            size *= dim
        return np.reshape(np_array, [-1,size])
