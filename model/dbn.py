import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def build_graph(self, graph, input_tensor, is_frozen=True,
                    activation='sigmoid', is_sampled=False):
        assert activation in ['sigmoid', 'relu']

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
                net = tf.matmul(input_tensor, weights) + bias
                if is_sampled:
                    rnd = tf.random_uniform(shape=tf.shape(net), name='rnd', dtype=tf.float32)
                    return tf.nn.relu(tf.sign(net-rnd))
                elif activation == 'sigmoid':
                    return tf.nn.sigmoid(net)
                elif activation == 'relu':
                    return tf.nn.relu(net)

    def build_gen_graph(self, graph, input_tensor, sample, k=0):
        with graph.as_default():
            with tf.name_scope(self.name+'_gen'):
                back_weights = tf.constant(self.W.T, name='weights_back', dtype=tf.float32)
                front_weights = tf.constant(self.W, name='weights_forward', dtype=tf.float32)

                bias_visible = tf.constant(self.bias_visible, name='bias_visible', dtype=tf.float32)
                bias_hidden = tf.constant(self.bias_hidden, name='bias_hidden', dtype=tf.float32)
                out = input_tensor

                for i in range(k):
                    out = tf.matmul(out, back_weights) + bias_visible # sample visible from hidden
                    if sample:
                        rnd = tf.random_uniform(out.shape, dtype=tf.float32)
                        out = tf.nn.relu(tf.sign(out - rnd)) # returns 0 or 1
                    else:
                        out = tf.nn.sigmoid(out)

                    out = tf.matmul(out, front_weights) + bias_hidden
                    if sample:
                        rnd = tf.random_uniform(out.shape, dtype=tf.float32)
                        out = tf.nn.relu(tf.sign(out - rnd)) # returns 0 or 1
                    else:
                        out = tf.nn.sigmoid(out)


                out = tf.matmul(out, back_weights) + bias_visible

                if sample:
                    rnd = tf.random_uniform(out.shape, dtype=tf.float32)
                    out = tf.nn.relu(tf.sign(out - rnd)) # returns 0 or 1
                else:
                    out = tf.nn.sigmoid(out)

                return out

class DBN:
    def __init__(self, hidden_layers, freeze_rbms, rbm_activation,
                 fully_connected_layers=[200],
                 connected_activations=['relu'],
                 output_activation='softmax',
                 keep_chance=0.9,
                 name='dbn'):
        self.rbms = []
        for l in hidden_layers:
            name = 'rbm_' + str(l)
            self.rbms.append(RBMLayer(l, name))

        self.fully_connected_layers = fully_connected_layers

        self.output_activation = output_activation
        self.name = name
        self.activations = connected_activations
        self.rbm_activation = rbm_activation
        self.freeze = freeze_rbms
        self.keep_chance = keep_chance
        self.graph = None

        self.rbm_out = None
        self.gen_in_placeholder = None
        self.gen_out = None

    def pretrain(self, train_set, pretrain_iterations=10000, learning_rate=0.01,
                 save_dir=None):
        train = train_set
        if len(train.shape) != 2:
            train = self.flatten_array(train)

        for i, layer in enumerate(self.rbms):
            best_loss = 999999.0
            since_improve = 0
            lr = learning_rate
            print('Pretraining layer', i+1, 'of', len(self.rbms))
            for j in tqdm(range(pretrain_iterations)):
                output = train[np.random.randint(0,train.shape[0],1)]
                for rbm in self.rbms[:i]:
                    output = rbm.sample_hidden_from_visible(output)

                loss = layer.RBMUpdate(output, lr)
                if loss < best_loss:
                    best_loss = loss
                    since_improve = 0
                elif since_improve > 1000:
                    lr *= 0.95
                    since_improve = 0
                else:
                    since_improve += 1

        print('Finished Pre-Training')

    def flatten_array(self, np_array):
        size = 1
        for dim in np_array.shape[1:]:
            size *= dim
        return np.reshape(np_array, [-1,size])

    def generate_example(self, train_set, train_labels, save_dir, out_dir, num_generations=10):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            model_saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init)

                res = []
                labels = []
                for i in range(num_generations):
                    example = np.random.randint(train_set.shape[0], size=1)
                    labels.append(train_labels[example])
                    train_image = train_set[example,:]
                    im_in = np.reshape(train_image, [28,28])
                    plt.imsave(os.path.join(out_dir, 'in'+str(i)+'.png'), im_in, cmap=plt.get_cmap('gray'))

                    forward = sess.run(self.rbm_out, feed_dict={self.in_placeholder:train_image, self.dropout_placeholder:1.0})
                    generated = sess.run(self.gen_out, feed_dict={self.gen_in_placeholder:forward})
                    im_gen = np.reshape(generated, [28,28])
                    plt.imsave(os.path.join(out_dir, 'gen'+str(i)+'.png'), im_gen, cmap=plt.get_cmap('gray'))

                return res, labels

    def generate_random_example(self, out_dir, num_generations):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                for i in range(num_generations):
                    generated = sess.run(self.random_gen_out)
                    im_gen = np.reshape(generated, [28,28])
                    plt.imsave(os.path.join(out_dir, 'gen'+str(i)+'.png'), im_gen, cmap=plt.get_cmap('gray'))

    def train(self, train_set, train_labels, validation_set, validation_labels,
              save_dir, learning_rate, decay_lr,  is_sampled, batch_size=100):
        if train_set.ndim != 2 or validation_set.ndim != 2:
            print('Training set and validation set must have dimension of 2')
            return None

        if train_labels.ndim != 2 or validation_labels.ndim != 2:
            print('Training and validation labels must have dimnsion of 2')
            return None

        self.build_graph(input_size=train_set.shape[1],
                         output_size=train_labels.shape[1],
                         learning_rate=learning_rate,
                         is_sampled=is_sampled)

        epochs = 0
        since_improved = 0
        best_loss = 999999.0
        loss_hist = []
        accuracy_hist = []
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            model_saver = tf.train.Saver()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with tf.Session() as sess:
                sess.run(init)
                while(since_improved < 10):
                    start = 0
                    train_set, train_labels = self.unison_shuffle(train_set, train_labels)
                    while(start + batch_size < train_set.shape[0]):
                        batch_data, batch_labels = train_set[start:start+batch_size], train_labels[start:start+batch_size]
                        start += batch_size
                        _ = sess.run([self.train_op],
                                      feed_dict={self.in_placeholder: batch_data,
                                                 self.out_placeholder: batch_labels,
                                                 self.dropout_placeholder: self.keep_chance})

                    loss_, acc = sess.run([self.loss,self.accuracy],
                                          feed_dict={self.in_placeholder: validation_set,
                                                     self.out_placeholder: validation_labels,
                                                     self.dropout_placeholder: 1.0})
                    loss_hist.append(loss_)
                    accuracy_hist.append(acc)
                    if loss_ < best_loss:
                        best_loss = loss_
                        since_improved = 0
                        model_saver.save(sess,
                                         os.path.join(save_dir, 'train.ckpt'),
                                         global_step=self.global_step)
                    else:
                        since_improved += 1

                    epochs += 1
                    print('Epoch', epochs,'\nBest Loss:', best_loss, '\nCur Validation Accuracy:', acc, '\n')

        print('finished training. Best validation loss = ', best_loss)
        return loss_hist, accuracy_hist

    def measure_test_accuracy(self, test_set, test_labels, save_dir):
        with self.graph.as_default():
            with tf.Session() as sess:
                model_loader = tf.train.Saver()
                init = tf.global_variables_initializer()
                sess.run(init)
                model_loader.restore(sess, tf.train.latest_checkpoint(save_dir))
                accuracy, loss = sess.run(
                                    [self.accuracy, self.loss],
                                    feed_dict={
                                        self.in_placeholder: test_set,
                                        self.out_placeholder: test_labels,
                                        self.dropout_placeholder: 1.0})

                return accuracy, loss

    def build_gen_graph(self, sample=False):
        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
            self.gen_in_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=[1,self.rbms[-1].num_hidden],
                                        name='gen_in')

            gen_in = self.gen_in_placeholder
            for i, rbm in enumerate(self.rbms[::-1]):
                if i == len(self.rbms) - 1:
                    gen_in = rbm.build_gen_graph(self.graph, gen_in, False)
                else:
                    gen_in = rbm.build_gen_graph(self.graph, gen_in, sample)

            self.gen_out = gen_in

    def build_random_gen_graph(self, k, sample=False):
        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
            gen_in = tf.random_uniform(
                            dtype=tf.float32,
                            shape=[1,self.rbms[-1].num_hidden],
                            name='random_in')
            last_rbm = self.rbms[-1]
            for i, rbm in enumerate(self.rbms[::-1]):
                if i == len(self.rbms) - 1:
                    gen_in = rbm.build_gen_graph(self.graph, gen_in, False)
                elif i == 0:
                    gen_in = rbm.build_gen_graph(self.graph, gen_in, sample,k)
                else:
                    gen_in = rbm.build_gen_graph(self.graph, gen_in, sample)

            self.random_gen_out = gen_in

    def build_graph(self, input_size, output_size, is_sampled=False, learning_rate=0.01):
        if self.graph is None:
            g = tf.Graph()
        else:
            g = self.graph

        with g.as_default():
            self.in_placeholder = tf.placeholder(tf.float32, [None, input_size])
            self.out_placeholder = tf.placeholder(tf.float32, [None, output_size])
            self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

            out = self.in_placeholder
            for rbm in self.rbms:
                out = rbm.build_graph(graph=g,
                                      input_tensor=out,
                                      is_frozen=self.freeze,
                                      activation=self.rbm_activation,
                                      is_sampled=is_sampled)
                out = tf.nn.dropout(out, self.dropout_placeholder)

            self.rbm_out = out
            if not self.rbms:
                num_prev_outputs = input_size
            else:
                num_prev_outputs = self.rbms[-1].num_hidden
            for i, connected in enumerate(self.fully_connected_layers):
                out = tf.nn.dropout(out, self.dropout_placeholder)
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

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            decay_learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                           100000, 0.96, staircase=True)
            self.train_op = tf.train.MomentumOptimizer(decay_learning_rate, momentum=0.9)
            self.train_op = self.train_op.minimize(
                                            self.loss,
                                            global_step=self.global_step)

        self.graph = g

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
