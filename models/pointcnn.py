import sys
sys.path.insert(0, './')

import math
from util import pointfly as pf
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout

from custom_layers.layers import XConv
from custom_layers.cpp_modules import farthest_point_sample


x = 8
xconv_param_name = ('K', 'D', 'P', 'C')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(12, 1, -1, 16 * x),
                 (16, 1, 768, 32 * x),
                 (16, 2, 384, 64 * x),
                 (16, 2, 128, 96 * x)]]

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 2, 3, 2),
                  (16, 1, 2, 1),
                  (12, 1, 1, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(16 * x, 0.0),
              (16 * x, 0.7)]]


class PointCNN(Model):

    def __init__(self, batch_size, num_classes, xconv_params, xdconv_params, fc_params,
                 X_transformation=True, sorting_method=None, sampling='fps', with_global=True, data_dim=6,):
        super(PointCNN, self).__init__()

        self.N = batch_size
        self.num_classes = num_classes
        self.xconv_params = xconv_params
        self.xdconv_params = xdconv_params
        self.fc_params = fc_params
        self.X_transformation = X_transformation
        self.sorting_method = sorting_method
        self.sampling = sampling
        self.with_global = with_global
        self.data_dim = data_dim
        self.xconv_list = []
        self.xdconv_list = []
        self.dense_list = []
        self.fc_list = []
        self.dropout_list = []

        self.init_network()

    def init_network(self):

        self.dense1 = pf.Dense(units=self.xconv_params[0]['C'] // 2, name='features_hd')

        for layer_idx, layer_param in enumerate(self.xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']

            if layer_idx == 0:
                C_pts_fts = C // 2 if self.data_dim == 3 else C // 4
                depth_multiplier = 4
            else:
                C_prev = self.xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (self.with_global and layer_idx == len(self.xconv_params) - 1)

            self.xconv_list.append(XConv(tag, self.N, K, D, P, C, C_pts_fts, self.X_transformation,
                                         depth_multiplier, self.sorting_method, with_global))

        for layer_idx, layer_param in enumerate(self.xdconv_params):
            tag = 'xdconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            pts_layer_idx = layer_param['pts_layer_idx']
            qrs_layer_idx = layer_param['qrs_layer_idx']
            P = self.xconv_params[qrs_layer_idx]['P']
            C = self.xconv_params[qrs_layer_idx]['C']
            C_prev = self.xconv_params[pts_layer_idx]['C']
            C_pts_fts = C_prev // 4
            depth_multiplier = 1

            self.xdconv_list.append(XConv(tag, self.N, K, D, P, C, C_pts_fts, self.X_transformation,
                                          depth_multiplier, self.sorting_method))
            self.dense_list.append(pf.Dense(units=C, name=tag + 'fts_fuse'))

        for layer_idx, layer_param in enumerate(self.fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            self.fc_list.append(pf.Dense(units=C, name='fc{:d}'.format(layer_idx)))
            self.dropout_list.append(Dropout(rate=dropout_rate, name='fc{:d}_drop'.format(layer_idx)))

        self.dense2 = pf.Dense(units=self.num_classes, name='output_layer',
                               activation=tf.nn.sigmoid if self.num_classes == 2 else tf.nn.softmax)

    def forward_pass(self, input, training):

        if self.data_dim == 3:
            layer_pts = [input]
            layer_fts = [None]
        else:
            layer_pts = [input[:, :, :3]]
            features = input[:, :, 3:]
            features = tf.reshape(features, (self.N, -1, self.data_dim - 3), name='features_reshape')
            layer_fts = [self.dense1(features, training=training)]

        for layer_idx, layer_param in enumerate(self.xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            P = layer_param['P']

            # get k-nearest points
            pts = layer_pts[-1]
            fts = layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == self.xconv_params[layer_idx - 1]['P']):
                qrs = layer_pts[-1]
            else:
                fps_indices = farthest_point_sample(P, pts)
                batch_indices = tf.tile(tf.reshape(tf.range(self.N), (-1, 1, 1)), (1, P, 1))
                indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                qrs = tf.gather_nd(pts, indices, name=tag + 'qrs') # (N, P, 3)
            layer_pts.append(qrs)

            fts_xconv = self.xconv_list[layer_idx](pts, fts, qrs, training=training)
            layer_fts.append(fts_xconv)
            # print("XConv " + str(layer_idx) + " points: " + str(qrs.shape) + " ---  features: " + str(fts_xconv.shape))

        for layer_idx, layer_param in enumerate(self.xdconv_params):
            tag = 'xdconv_' + str(layer_idx + 1) + '_'
            pts_layer_idx = layer_param['pts_layer_idx']
            qrs_layer_idx = layer_param['qrs_layer_idx']

            pts = layer_pts[pts_layer_idx + 1]
            fts = layer_fts[pts_layer_idx + 1] if layer_idx == 0 else layer_fts[-1]
            qrs = layer_pts[qrs_layer_idx + 1]
            fts_qrs = layer_fts[qrs_layer_idx + 1]

            fts_xdconv = self.xdconv_list[layer_idx](pts, fts, qrs, training=training)

            fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')

            fts_fuse = self.dense_list[layer_idx](fts_concat)
            layer_pts.append(qrs)
            layer_fts.append(fts_fuse)
            # print("XDe-Conv " + str(layer_idx) + " points: " + str(qrs.shape) + " ---  features: " + str(fts_fuse.shape))

        net = layer_fts[-1]
        for fc_layer, dropout in zip(self.fc_list, self.dropout_list):
            net = fc_layer(net)
            net = dropout(net)

        pred = self.dense2(net)

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
