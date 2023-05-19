import sys
sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout

from custom_layers.utils import Conv2d
from custom_layers.layers import SpatialTransform, EdgeConv
from util.dgnet_util import pairwise_distance, knn, get_edge_feature


class DGCNN(Model):

    def __init__(self, batch_size, num_classes, num_points, activation=tf.nn.relu, bn=True):
        super(DGCNN, self).__init__()

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_points = num_points
        self.keep_prob = 0.7
        self.k = 20
        self.activation = activation
        self.bn = bn

        self.init_network()

    def init_network(self):

        self.transform_net = SpatialTransform(self.num_points, conv=[64, 128, 1024],
                                              mlp=[512, 256], k=3,
                                              activation=self.activation, bn=self.bn)

        self.edge_conv1 = EdgeConv([64, 64], activation=self.activation, bn=self.bn)
        self.edge_conv2 = EdgeConv([64, 64], activation=self.activation, bn=self.bn)
        self.edge_conv3 = EdgeConv([64], activation=self.activation, bn=self.bn)

        self.conv1 = Conv2d(1024, activation=self.activation, bn=self.bn)

        self.maxpool2d = tf.keras.layers.MaxPool2D(pool_size=[self.num_points, 1],
                                                   strides=[2, 2])

        self.conv2 = Conv2d(512, activation=self.activation, bn=self.bn)
        self.conv3 = Conv2d(256, activation=self.activation, bn=self.bn)
        self.dropout = Dropout(self.keep_prob)
        self.conv4 = Conv2d(self.num_classes, activation=None, bn=False)

    def forward_pass(self, input, training):

        input = input[:, :, :3]
        input_image = tf.expand_dims(input, -1)

        adj = pairwise_distance(input)
        nn_idx = knn(adj, k=self.k)
        edge_feature = get_edge_feature(input_image, nn_idx=nn_idx, k=self.k)

        transform = self.transform_net(edge_feature, training=training)
        input_transformed = tf.matmul(input, transform)

        input_image = tf.expand_dims(input_transformed, -1)
        adj = pairwise_distance(input_transformed)
        nn_idx = knn(adj, k=self.k)
        edge_feature = get_edge_feature(input_image, nn_idx=nn_idx, k=self.k)

        x1 = self.edge_conv1(edge_feature)

        adj = pairwise_distance(x1)
        nn_idx = knn(adj, k=self.k)
        edge_feature = get_edge_feature(x1, nn_idx=nn_idx, k=self.k)

        x2 = self.edge_conv2(edge_feature)

        adj = pairwise_distance(x2)
        nn_idx = knn(adj, k=self.k)
        edge_feature = get_edge_feature(x2, nn_idx=nn_idx, k=self.k)

        x3 = self.edge_conv3(edge_feature)

        net = self.conv1(tf.concat([x1, x2, x3], axis=-1))
        net = self.maxpool2d(net)

        expand = tf.tile(net, [1, self.num_points, 1, 1])
        net = tf.concat(axis=3, values=[expand, x1, x2, x3])

        net = self.conv2(net)
        net = self.conv3(net)
        net = self.dropout(net)

        net = self.conv4(net)
        pred = tf.squeeze(net, [2])

        return pred

    def train_step(self, input):

        with tf.GradientTape() as tape:

            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):

        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, input, training=False):

        return self.forward_pass(input, training)








