import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from . import utils
from util import pointfly as pf


class Pointnet_SA(Layer):

	def __init__(
		self, npoint, radius, nsample, mlp, group_all=False, knn=False, use_xyz=True, activation=tf.nn.relu, bn=False
	):

		super(Pointnet_SA, self).__init__()

		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.mlp = mlp
		self.group_all = group_all
		self.knn = False
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn

		self.mlp_list = []

	def build(self, input_shape):

		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

		super(Pointnet_SA, self).build(input_shape)

	def call(self, xyz, points, training=True):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)

		if self.group_all:
			nsample = xyz.get_shape()[1]
			new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group_all(xyz, points, self.use_xyz)
		else:
			new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group(
				self.npoint,
				self.radius,
				self.nsample,
				xyz,
				points,
				self.knn,
				use_xyz=self.use_xyz
			)

		for i, mlp_layer in enumerate(self.mlp_list):
			new_points = mlp_layer(new_points, training=training)

		new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

		return new_xyz, tf.squeeze(new_points)


class Pointnet_SA_MSG(Layer):

	def __init__(
		self, npoint, radius_list, nsample_list, mlp, use_xyz=True, activation=tf.nn.relu, bn = False
	):

		super(Pointnet_SA_MSG, self).__init__()

		self.npoint = npoint
		self.radius_list = radius_list
		self.nsample_list = nsample_list
		self.mlp = mlp
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn

		self.mlp_list = []

	def build(self, input_shape):

		for i in range(len(self.radius_list)):
			tmp_list = []
			for i, n_filters in enumerate(self.mlp[i]):
				tmp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))
			self.mlp_list.append(tmp_list)

		super(Pointnet_SA_MSG, self).build(input_shape)

	def call(self, xyz, points, training=True):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)

		new_xyz = utils.gather_point(xyz, utils.farthest_point_sample(self.npoint, xyz))

		new_points_list = []

		for i in range(len(self.radius_list)):
			radius = self.radius_list[i]
			nsample = self.nsample_list[i]
			idx, pts_cnt = utils.query_ball_point(radius, nsample, xyz, new_xyz)
			grouped_xyz = utils.group_point(xyz, idx)
			grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])

			if points is not None:
				grouped_points = utils.group_point(points, idx)
				if self.use_xyz:
					grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
			else:
				grouped_points = grouped_xyz

			for i, mlp_layer in enumerate(self.mlp_list[i]):
				grouped_points = mlp_layer(grouped_points, training=training)

			new_points = tf.math.reduce_max(grouped_points, axis=2)
			new_points_list.append(new_points)

		new_points_concat = tf.concat(new_points_list, axis=-1)

		return new_xyz, new_points_concat


class Pointnet_FP(Layer):

	def __init__(
		self, mlp, activation=tf.nn.relu, bn=False
	):

		super(Pointnet_FP, self).__init__()

		self.mlp = mlp
		self.activation = activation
		self.bn = bn

		self.mlp_list = []

	def build(self, input_shape):

		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))
		super(Pointnet_FP, self).build(input_shape)

	def call(self, xyz1, xyz2, points1, points2, training=True):

		if points1 is not None:
			if len(points1.shape) < 3:
				points1 = tf.expand_dims(points1, axis=0)
		if points2 is not None:
			if len(points2.shape) < 3:
				points2 = tf.expand_dims(points2, axis=0)

		dist, idx = utils.three_nn(xyz1, xyz2)
		dist = tf.maximum(dist, 1e-10)
		norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
		norm = tf.tile(norm,[1,1,3])
		weight = (1.0/dist) / norm
		interpolated_points = utils.three_interpolate(points2, idx, weight)

		if points1 is not None:
			new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
		else:
			new_points1 = interpolated_points
		new_points1 = tf.expand_dims(new_points1, 2)

		for i, mlp_layer in enumerate(self.mlp_list):
			new_points1 = mlp_layer(new_points1, training=training)

		new_points1 = tf.squeeze(new_points1)
		if len(new_points1.shape) < 3:
			new_points1 = tf.expand_dims(new_points1, axis=0)

		return new_points1


class XConv(Layer):

	def __init__(
			self, tag, N, K, D, P, C, C_pts_fts, X_transformation, depth_multiplier,
			sorting_method=None, with_global=False
	):
		super(XConv, self).__init__()

		self.tag = tag
		self.N = N
		self.K = K
		self.D = D
		self.P = P
		self.C = C
		self.C_pts_fts = C_pts_fts
		self.X_transformation = X_transformation
		self.depth_multiplier = depth_multiplier
		self.sorting_method = sorting_method
		self.with_global = with_global

	def build(self, input_shape):

		self.dense1 = pf.Dense(units=self.C_pts_fts, name=self.tag + 'nn_fts_from_pts_0')
		self.dense2 = pf.Dense(units=self.C_pts_fts, name=self.tag + 'nn_pts_local')

		self.conv1 = pf.Conv2D(filters=self.K * self.K, name=self.tag + 'X_0', kernel_size=(1, self.K))
		self.dconv1 = pf.Depthwise_Conv2D(
			filters=self.K * self.K, depth_multiplier=self.K, name=self.tag + 'X_1', kernel_size=(1, self.K))
		self.dconv2 = pf.Depthwise_Conv2D(
			filters=self.K * self.K, depth_multiplier=self.K, name=self.tag + 'X_1', kernel_size=(1, self.K), activation=None)

		self.sconv1 = pf.Seperable_Conv2D(
			filters=self.C, name=self.tag + 'fts_conv', kernel_size=(1, self.K), depth_multiplier=self.depth_multiplier)

		self.dense3 = pf.Dense(units=self.C // 4, name=self.tag + 'fts_global_0')
		self.dense4 = pf.Dense(units=self.C // 4, name=self.tag + 'fts_global')

		super(XConv, self).build(input_shape)

	def call(self, pts, fts, qrs, training=True):

		_, indices_dilated = pf.knn_indices_general(qrs, pts, self.K * self.D, True)
		indices = indices_dilated[:, :, ::self.D, :]

		if self.sorting_method is not None:
			indices = pf.sort_points(pts, indices, self.sorting_method)

		nn_pts = tf.gather_nd(pts, indices, name=self.tag + 'nn_pts')  # (N, P, K, 3)
		nn_pts_center = tf.expand_dims(qrs, axis=2, name=self.tag + 'nn_pts_center')  # (N, P, 1, 3)
		nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=self.tag + 'nn_pts_local')  # (N, P, K, 3)

		# Prepare features to be transformed
		nn_fts_from_pts_0 = self.dense1(nn_pts_local, training=training)
		nn_fts_from_pts = self.dense2(nn_fts_from_pts_0, training=training)

		if fts is None:
			nn_fts_input = nn_fts_from_pts
		else:
			nn_fts_from_prev = tf.gather_nd(fts, indices, name=self.tag + 'nn_fts_from_prev')
			nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=self.tag + 'nn_fts_input')

		if self.X_transformation:
			######################## X-transformation #########################
			X_0 = self.conv1(nn_pts_local, training=training)
			X_0_KK = tf.reshape(X_0, (self.N, self.P, self.K, self.K), name=self.tag + 'X_0_KK')
			X_1 = self.dconv1(X_0_KK, training=training)
			X_1_KK = tf.reshape(X_1, (self.N, self.P, self.K, self.K), name=self.tag + 'X_1_KK')
			X_2 = self.dconv2(X_1_KK, training=training)
			X_2_KK = tf.reshape(X_2, (self.N, self.P, self.K, self.K), name=self.tag + 'X_2_KK')
			fts_X = tf.matmul(X_2_KK, nn_fts_input, name=self.tag + 'fts_X')
			###################################################################
		else:
			fts_X = nn_fts_input

		fts_conv = self.sconv1(fts_X, training=training)
		fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=self.tag + 'fts_conv_3d')

		if self.with_global:
			fts_global_0 = self.dense3(qrs, training=training)
			fts_global = self.dense4(fts_global_0, training=training)
			return tf.concat([fts_global, fts_conv_3d], axis=-1, name=self.tag + 'fts_conv_3d_with_global')
		else:
			return fts_conv_3d


class SpatialTransform(Layer):

	def __init__(self, num_points, conv=[64, 128, 1024], mlp=[512, 256], k=3, activation=tf.nn.relu, bn=True, l2reg=.001):
		super(SpatialTransform, self).__init__()

		self.num_points = num_points
		self.k = k
		self.activation = activation
		self.bn = bn
		self.conv = conv
		self.mlp = mlp
		self.bias = tf.keras.initializers.Constant(np.eye(k).flatten())
		self.reg = self.OrthogonalRegularizer(k, l2reg)

		self.mlp_list = []
		self.conv_list = []

	class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
		def __init__(self, num_features, l2reg):
			self.num_features = num_features
			self.l2reg = l2reg
			self.eye = tf.eye(num_features)

		def __call__(self, x):
			x = tf.reshape(x, (-1, self.num_features, self.num_features))
			xxt = tf.tensordot(x, x, axes=(2, 2))
			xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
			return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

	def build(self, input_shape):

		for n_filters in self.conv:
			self.conv_list.append(
				utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

		for n_units in self.mlp:
			self.mlp_list.append(utils.Dense(
				n_units, activation=self.activation, bn=self.bn))

		self.maxpool2d = tf.keras.layers.MaxPool2D(
			pool_size=[self.num_points, 1], strides=[2, 2])

		self.dense = tf.keras.layers.Dense(
			self.k * self.k,
			kernel_initializer="zeros",
			bias_initializer=self.bias,
			activity_regularizer=self.reg)

	def call(self, inputs, training=True):

		batch_size = inputs.shape[0]

		net = inputs
		for conv_layer in self.conv_list[:-1]:
			net = conv_layer(net, training=training)

		net = tf.reduce_max(net, axis=-2, keepdims=True)

		net = self.conv_list[-1](net, training=training)
		net = self.maxpool2d(net)
		net = tf.reshape(net, [batch_size, -1])

		for mlp_layer in self.mlp_list:
			net = mlp_layer(net, training=training)
		net = self.dense(net)

		return tf.keras.layers.Reshape((self.k, self.k))(net)


class EdgeConv(Layer):

	def __init__(self, filters_list, activation=tf.nn.relu, bn=True):
		super(EdgeConv, self).__init__()

		self.filters_list = filters_list
		self.activation = activation
		self.bn = bn

		self.conv_list = []

	def build(self, input_shape):

		for n_filters in self.filters_list:
			self.conv_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

	def call(self, inputs, training=True):

		edge_feature = inputs

		for conv_layer in self.conv_list:
			edge_feature = conv_layer(edge_feature, training=training)

		return tf.reduce_max(edge_feature, axis=-2, keepdims=True)





