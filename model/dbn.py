import numpy as np
import tensorflow as tf

def make_full_layer(graph, name, num_inputs, input_tensor, num_units, activation):
    assert activation in ['relu', 'sigmoid', 'softmax']
    with graph.as_default():
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_units], stddev=0.1), name='weights')
            biases = tf.Variable(tf.zeros([num_units]), name='bias')
            net = tf.matmul(input_tensor, weights) + biases;
            if activation == 'relu':
                return tf.nn.relu(net), net
            elif activation == 'sigmoid':
                return tf.nn.sigmoid(net), net
            else:
                return tf.nn.softmax(net), net

class RBMLayer:
    def __init__(self, num_hidden, name):
        self.W = None
        self.bias_visible = None
        self.bias_hidden = np.zeros([1, num_hidden])

        self.num_hidden = num_hidden
        self.name = name

        # Built during build_graph
        self.graph = None
        self.accuracy = None
        self.loss = None
        self.train_op = None

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
        delta_W = learning_rate * (np.dot(h1.T, x1) - np.dot(h_prob.T, x2))
        self.bias_hidden += learning_rate * (h1 - h_prob)
        self.bias_visible += learning_rate * (x1 - x2)
        self.W += delta_W.T
        return np.sum(np.absolute(delta_W))

    def build_graph(self, graph, input_tensor, is_frozen=True, is_sampled=False):
        with graph.as_default():
            with tf.name_scope(self.name):
                if is_frozen:
                    weights = tf.constant(self.W, name='weights', dtype=tf.float32)
                else:
                    weights = tf.Variable(self.W, name='weights', dtype=tf.float32)

                bias = tf.constant(self.bias_hidden, name='bias', dtype=tf.float32)
                '''if is_sampled:
                    return tf.nn.relu()
                else:'''
                return tf.nn.sigmoid(tf.matmul(input_tensor, weights) + bias)

class DBN:
    def __init__(self, hidden_layers, fully_connected_layers=[200],
                 connected_activations=['relu'], output_activation='softmax',
                 name='dbn'):
        self.rbms = []
        for l in hidden_layers:
            name = 'rbm_' + str(l)
            self.rbms.append(RBMLayer(l, name))

        self.fully_connected_layers = fully_connected_layers

        self.output_activation = output_activation
        self.name = name
        self.activations = connected_activations

    def pretrain(self, train_set, learning_rate=0.01):
        train = train_set
        if len(train.shape) != 2:
            train = self.flatten_array(train)

        for i, layer in enumerate(self.rbms):
            best_loss = 999999.0
            since_improve = 0
            lr = learning_rate
            print('Pretraining layer', i+1, 'of', len(self.rbms))
            for j in range(100):
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

    def train(self, train_set, train_labels, validation_set, validation_labels, batch_size=100):
        if train_set.ndim != 2 or validation_set.ndim != 2:
            print('Training set and validation set must have dimension of 2')
            return None

        if train_labels.ndim != 2 or validation_labels.ndim != 2:
            print('Training and validation labels must have dimnsion of 2')
            return None

        self.build_graph(train_set.shape[1], train_labels.shape[1])
        print(self.loss.shape)
        epochs = 0
        since_improved = 0
        best_loss = 999999.0
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                while(since_improved < 5):
                    start = 0
                    train_set, train_labels = self.unison_shuffle(train_set, train_labels)
                    while(start + batch_size < train_set.shape[0]):
                        batch_data, batch_labels = train_set[start:start+batch_size], train_labels[start:start+batch_size]
                        start += batch_size
                        _ = sess.run([self.train_op],
                                      feed_dict={self.in_placeholder: batch_data,
                                                 self.out_placeholder: batch_labels})

                    loss_ = sess.run(self.loss, feed_dict={self.in_placeholder: validation_set,
                                                        self.out_placeholder: validation_labels})
                    if loss_ < best_loss:
                        best_loss = loss_
                        since_improved = 0
                    else:
                        since_improved += 1

                    epochs += 1
                    print(best_loss)

        print('finished training. Best loss = ', best_loss)


    def build_graph(self, input_size, output_size):
        g = tf.Graph()
        with g.as_default():
            self.in_placeholder = tf.placeholder(tf.float32, [None, input_size])
            self.out_placeholder = tf.placeholder(tf.float32, [None, output_size])

            out = self.in_placeholder
            for rbm in self.rbms:
                out = rbm.build_graph(graph=g, input_tensor=out)

            num_prev_outputs = self.rbms[-1].num_hidden
            for i, connected in enumerate(self.fully_connected_layers):
                out, _ = make_full_layer(
                        graph=g, name='fully_connected_'+str(i),
                        num_inputs=num_prev_outputs, num_units=connected,
                        activation=self.activations[i],
                        input_tensor=out)
                num_prev_outputs = connected

            out, net = make_full_layer(
                    graph=g, name='softmax_output',num_inputs=num_prev_outputs,
                    num_units=output_size, activation='softmax',
                    input_tensor=out)

            with tf.name_scope('loss'):
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=net, labels=self.out_placeholder, name='cross_entropy')
                self.loss = tf.reduce_mean(loss, name='xentropy_mean')

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(self.out_placeholder, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss, global_step=global_step)

        self.graph = g

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
