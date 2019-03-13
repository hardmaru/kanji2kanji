## Kanji2Kanji library

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import SVG, display

import numpy as np

from scipy.misc import imresize as resize
from scipy.misc import imsave

import svgwrite # conda install -c omnia svgwrite=1.1.6
import os
import json
import random
import time

import requests

import tensorflow as tf
from tensorflow.python.util import nest as tf_nest

from numba import jit
from numpy import arange

np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# default settings:

Z_SIZE = 64
RNN_BATCH_SIZE = 50

IMAGE_SIZE = 64

MAX_LEN = 134
CONDITIONAL_MODE = 1

RNN_SIZE = 1024
NUM_MIXTURE = 50
NUM_LAYERS = 3
USE_DROPOUT = 1
DROPOUT_KEEP_PROB = 0.3

# tensorflow modules
def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hps_train = tf.contrib.training.HParams(
                     max_seq_len=MAX_LEN, # train on sequences of 100
                     seq_width=2,    # width of our data (x and y)
                     rnn_size=RNN_SIZE,    # number of rnn cells
                     num_layers=NUM_LAYERS,
                     batch_size=RNN_BATCH_SIZE,   # minibatch sizes
                     z_size=Z_SIZE,
                     grad_clip=1.0,
                     num_mixture=NUM_MIXTURE,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=USE_DROPOUT,
                     recurrent_dropout_prob=DROPOUT_KEEP_PROB,
                     use_input_dropout=0,
                     input_dropout_prob=DROPOUT_KEEP_PROB,
                     use_output_dropout=0,
                     output_dropout_prob=DROPOUT_KEEP_PROB,
                     conditional=CONDITIONAL_MODE,
                     is_training=1
  )

  hps_test = copy_hparams(hps_train)
  hps_test.use_recurrent_dropout=0
  hps_test.use_input_dropout=0
  hps_test.use_output_dropout=0
  hps_test.is_training=0
  hps_sample = copy_hparams(hps_test)
  hps_sample.batch_size=1
  hps_sample.max_seq_len=1
  return hps_train, hps_test, hps_sample

def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())

def default_hps():
  return get_default_hparams()

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

def initial_cell_state_from_embedding(cell, z, name=None):
  """Computes an initial RNN `cell` state from an embedding, `z`."""
  flat_state_sizes = tf_nest.flatten(cell.state_size)
  return tf_nest.pack_sequence_as(
    cell.zero_state(batch_size=z.shape[0], dtype=tf.float32),
    tf.split(
      tf.layers.dense(
        z,
        sum(flat_state_sizes),
        activation=tf.tanh,
        name=name),
      flat_state_sizes,
      axis=1))

# MDN-RNN model tailored for quickdraw, with multiple categories
# MDN-RNN model tailored for quickdraw, with multiple categories
class MDNRNN():
  def __init__(self, hps, gpu_mode=True, layer_norm=False, reuse=False):
    self.hps = hps
    self.layer_norm = layer_norm
    with tf.variable_scope('mdn_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu")
        self.g = tf.Graph()
        with self.g.as_default():
          self.build_model(hps)
    self.init_session()
  def build_model(self, hps):

    self.num_mixture = hps.num_mixture
    KMIX = self.num_mixture # 5 mixtures
    WIDTH = hps.seq_width # 2 channels
    LENGTH = self.hps.max_seq_len+1 # 101 steps

    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    layer_cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
    basic_cell_fn = tf.contrib.rnn.BasicLSTMCell
    
    if self.layer_norm:
      basic_cell_fn = layer_cell_fn
    
    num_layers = self.hps.num_layers

    use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True
    use_layer_norm = False if self.hps.use_layer_norm == 0 else True
    is_training = False if self.hps.is_training == 0 else True

    def rnn_cell():
      if use_recurrent_dropout:
        cell = layer_cell_fn(hps.rnn_size,
                     layer_norm=use_layer_norm,
                     dropout_keep_prob=self.hps.recurrent_dropout_prob)
      elif use_layer_norm:
        cell = layer_cell_fn(hps.rnn_size,
                     layer_norm=use_layer_norm)
      else:
        if self.layer_norm:
          cell = layer_cell_fn(hps.rnn_size, layer_norm=use_layer_norm)
        else:
          cell = basic_cell_fn(hps.rnn_size)
      if use_input_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
      return cell

    # multi-layer, and dropout:
    print("layer norm mode =", use_layer_norm)
    print("layers =", num_layers)
    print("input dropout mode =", use_input_dropout)
    print("output dropout mode =", use_output_dropout)
    print("recurrent dropout mode =", use_recurrent_dropout)
    if use_input_dropout:
      print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)

    # multi layer lstm
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)], state_is_tuple=True)
    else:
      cell = rnn_cell()
    if use_output_dropout:
      print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, LENGTH, 5])

    self.target_x = self.sequence[:, 1:, :WIDTH] # x and y floats
    self.target_pen = self.sequence[:, 1:, WIDTH:] # one-hot of 3 states
    
    self.input_seq = self.sequence[:, :LENGTH-1, :]

    self.zero_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)

    self.batch_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.z_size])

    # tile conditioning vector over each timestep of input
    pre_tile_condition = tf.reshape(self.batch_z,
                            [self.hps.batch_size, 1, self.hps.z_size])

    overlay_seq = tf.tile(pre_tile_condition, [1, self.hps.max_seq_len, 1])
    actual_input_seq = tf.concat([self.input_seq, overlay_seq], axis=2)

    self.initial_state = initial_cell_state_from_embedding(
      cell,
      self.batch_z,
      name='decoder/z_to_initial_state')

    NOUT = WIDTH * KMIX * 3 + 3 # mixture of gaussians plus 3 extra states

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    output, last_state = tf.nn.dynamic_rnn(cell, actual_input_seq, initial_state=self.initial_state,
                                           time_major=False, swap_memory=True, dtype=tf.float32, scope="RNN")

    output = tf.reshape(output, [-1, hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)

    self.out_pen_logits = output[:, :3]
    output = output[:, 3:]

    output = tf.reshape(output, [-1, KMIX * 3])
    self.final_state = last_state    

    logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

    def tf_lognormal(y, mean, logstd):
      return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def get_lossfunc(logmix, mean, logstd, y, pen_data):
      v = logmix + tf_lognormal(y, mean, logstd)
      v = tf.reduce_logsumexp(v, 1, keepdims=True)
      v = tf.reshape(v, [hps.batch_size, hps.max_seq_len, WIDTH])

      fs = 1.0 - pen_data[:, :, 2]  # use training data for this
      fs = tf.expand_dims(fs, -1)
      # Zero out loss terms beyond N_s, the last actual stroke
      v = tf.multiply(v, fs)
      return -tf.reduce_mean(tf.reduce_sum(v, axis=2))

    def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
      return logmix, mean, logstd

    out_logmix, out_mean, out_logstd = get_mdn_coef(output)

    self.out_logmix = out_logmix
    self.out_mean = out_mean
    self.out_logstd = out_logstd

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_x,[-1, 1])
    flat_target_pen = tf.reshape(self.target_pen, [-1, 3])

    self.stroke_cost = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data, self.target_pen)

    self.pen_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=flat_target_pen,
                                                          logits=tf.reshape(self.out_pen_logits,[-1, 3])))
    
    self.cost = self.stroke_cost + self.pen_cost

    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      gvs = optimizer.compute_gradients(self.cost)
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    # initialize vars
    self.init = tf.global_variables_initializer()

  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def save_model(self, model_save_path, epoch):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'sketchrnn')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, epoch) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1
  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='rnn.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

class Latent2Latent(object):
  def __init__(self, batch_size=100, hidden_size=1024, learning_rate=0.001, is_training=True, reuse=False, gpu_mode=True):
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.hidden_size = hidden_size
    self.is_training = is_training
    self.reuse = reuse
    with tf.variable_scope('Image2Image', reuse=self.reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()
  def _build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():

      self.x = tf.placeholder(tf.float32, shape=[None, Z_SIZE])
      self.y = tf.placeholder(tf.float32, shape=[None, Z_SIZE])
      
      h = tf.layers.dense(self.x, self.hidden_size, activation=tf.nn.leaky_relu, name="input_layer")
      # Encoder
      for i in range(2):
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.leaky_relu, name="hidden_layer_"+str(i))

      #self.predict = tf.layers.dense(h, Z_SIZE, activation=None, name="output_layer")
      
      # MDN start
      KMIX = 100

      NOUT = Z_SIZE * KMIX * 3
      output = tf.layers.dense(h, NOUT, activation=None, name="output_layer")
      output = tf.reshape(output, [-1, KMIX * 3])

      logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

      def tf_lognormal(y, mean, logstd):
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

      def get_lossfunc(logmix, mean, logstd, y):
        v = logmix + tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, 1, keepdims=True)
        return -tf.reduce_mean(v)

      def get_mdn_coef(output):
        logmix, mean, logstd = tf.split(output, 3, 1)
        logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

      out_logmix, out_mean, out_logstd = get_mdn_coef(output)

      self.out_logmix = tf.reshape(out_logmix, [-1, Z_SIZE, KMIX])
      self.out_mean = tf.reshape(out_mean, [-1, Z_SIZE, KMIX])
      self.out_logstd = tf.reshape(out_logstd, [-1, Z_SIZE, KMIX])

      # MDN end

      # train ops
      if self.is_training:
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # MDN loss
        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.y, [-1, 1])
        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)
        self.loss = tf.reduce_mean(lossfunc)

        # training
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

        self.train_op = self.optimizer.apply_gradients(
          grads, global_step=self.global_step, name='train_step')

      # initialize vars
      self.init = tf.global_variables_initializer()

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def translate(self, x):
    return self.sess.run(self.predict, feed_dict={self.x: x})
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'image2image')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

class ConvVAE(object):
  def __init__(self, z_size=256, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True, reuse=False, gpu_mode=True):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    with tf.variable_scope('conv_vae', reuse=self.reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()
  def _build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():

      self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
      self.y = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
      
      mf = 1

      # Encoder
      h = tf.layers.conv2d(self.x, 32*mf, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
      h = tf.layers.conv2d(h, 64*mf, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
      h = tf.layers.conv2d(h, 128*mf, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
      h = tf.layers.conv2d(h, 256*mf, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
      h = tf.reshape(h, [-1, 2*2*256*mf])

      # VAE
      #'''
      self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
      self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
      self.sigma = tf.exp(self.logvar / 2.0)
      self.epsilon = tf.random_normal([self.batch_size, self.z_size])
      self.z = self.mu + self.sigma * self.epsilon
      #'''
      #self.z = h

      # Decoder
      h = tf.layers.dense(self.z, 4*256*mf, name="dec_fc")
      #h = self.z
      h = tf.reshape(h, [-1, 1, 1, 4*256*mf])
      h = tf.layers.conv2d_transpose(h, 128*mf, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
      h = tf.layers.conv2d_transpose(h, 64*mf, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
      h = tf.layers.conv2d_transpose(h, 32*mf, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
      self.predict = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")

      # train ops
      if self.is_training:
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        eps = 1e-6 # avoid taking log of zero

        # reconstruction loss
        # L1 loss
        '''
        self.r_loss = tf.reduce_sum(
          tf.abs(self.y - self.predict),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)
        '''

        # L2 loss
        '''
        self.r_loss = tf.reduce_sum(
          tf.square(self.y - self.predict),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)
        '''

        # log loss
        #'''
        self.r_loss = - tf.reduce_mean(
          self.y * tf.log(self.predict + eps) + (1.0 - self.y) * tf.log(1.0 - self.predict + eps),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)*64.0*64.0
        #'''

        # augmented kl loss per dim
        #'''
        self.kl_loss = - 0.5 * tf.reduce_sum(
          (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
          reduction_indices = 1
        )
        self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
        self.kl_loss = tf.reduce_mean(self.kl_loss)

        self.loss = self.r_loss + self.kl_loss
        #'''
        #self.loss = self.r_loss

        # training
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

        self.train_op = self.optimizer.apply_gradients(
          grads, global_step=self.global_step, name='train_step')

      # initialize vars
      self.init = tf.global_variables_initializer()

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})
  def encode_mu(self, x):
    return self.sess.run(self.mu, feed_dict={self.x: x})
  def decode(self, z):
    return self.sess.run(self.predict, feed_dict={self.z: z})
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

def load_data(sketch_rnn_mode=False):
  load_data = np.load('data/old_kanji.npz', encoding='latin1')
  old_train_category = load_data['category']
  old_train_data = load_data['data']
  old_train_label = load_data['label']

  load_data = np.load('data/new_kanji_train.npz', encoding='latin1')
  new_train_category = load_data['category']
  new_train_data = load_data['data']
  new_train_label = load_data['label']

  overlap_unicode = []
  for label in old_train_category:
    if label in new_train_category:
      overlap_unicode.append(label)

  old_kanji_data = {}
  old_kanji_data_train = {}
  old_kanji_data_test = {}
  for label in overlap_unicode:
    old_kanji_data[label] = []
    old_kanji_data_train[label] = []
    old_kanji_data_test[label] = []
  for i in range(len(old_train_label)):
    label = old_train_label[i]
    if label in overlap_unicode:
      old_kanji_data[label].append(old_train_data[i])
  test_unicode = []
  for label in overlap_unicode:
    if len(old_kanji_data[label]) > 1:
      old_kanji_data_test[label].append(old_kanji_data[label][0])
      test_unicode.append(label)
      old_kanji_data_train[label] = old_kanji_data[label][1:]
    else:
      old_kanji_data_train[label] = old_kanji_data[label]

  new_kanji_data = {}
  
  if sketch_rnn_mode:
    for label in new_train_category:
      new_kanji_data[label] = []
    for i in range(len(new_train_label)):
      label = new_train_label[i]
      new_kanji_data[label].append(new_train_data[i])
  else:
    for label in overlap_unicode:
      new_kanji_data[label] = []
    for i in range(len(new_train_label)):
      label = new_train_label[i]
      if label in overlap_unicode:
        new_kanji_data[label].append(new_train_data[i])
  return overlap_unicode, test_unicode, old_kanji_data_train, old_kanji_data_test, new_kanji_data

def augment_pixel(pixel_image, size_low = 48, size_high=64):
  dim_1 = np.random.randint(size_low, size_high+1)
  dim_2 = np.random.randint(size_low, size_high+1)
  resized_image = resize(pixel_image, (dim_1, dim_2), interp='lanczos')
  start_1 = np.random.randint(0, size_high-dim_1+1)
  start_2 = np.random.randint(0, size_high-dim_2+1)
  result = np.copy(pixel_image) * 0
  result[start_1:start_1+dim_1, start_2:start_2+dim_2] = resized_image
  return result

def random_batch(batch_size, labels, old_data, new_data):
  idx = np.random.permutation(range(0, len(labels)))[0:batch_size]
  x = []
  y = []
  u = []
  for i in idx:
    label = labels[i]
    u.append(label)
    x.append(augment_pixel(random.choice(old_data[label])).astype(np.float32)/255.0)
    y.append(augment_pixel(random.choice(new_data[label])).astype(np.float32)/255.0)
  x = np.array(x).reshape(batch_size, 64, 64, 1)
  y = np.array(y).reshape(batch_size, 64, 64, 1)
  u = np.array(u, dtype=np.int32)
  return x, y, u

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.045, svg_filename = '/tmp/sketch_rnn/svg/sample.svg', display_mode=True):
  if not os.path.exists(os.path.dirname(svg_filename)):
    os.makedirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  if display_mode:
    display(SVG(dwg.tostring()))


def show_image(img):
  plt.imshow(1-img.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
  plt.show()

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=5.0, grid_space_x=5.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = np.array([[0, 0, 1]] + sample[0].tolist())
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 1])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

def read_categories(filename):
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip() for x in content]
  return content

def get_bounds(data, factor=1):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
  """Spherical interpolation."""
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
  """Linear interpolation."""
  return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  for i in range(len(strokes)):
    if strokes[i, 2] == 1:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
      lines.append(line)
      line = []
    else:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
  return lines


def lines_to_strokes(lines):
  """Convert polyline format to stroke-3 format."""
  eos = 0
  strokes = [[0, 0, 0]]
  for line in lines:
    linelen = len(line)
    for i in range(linelen):
      eos = 0 if i < linelen - 1 else 1
      strokes.append([line[i][0], line[i][1], eos])
  strokes = np.array(strokes)
  strokes[1:, 0:2] -= strokes[:-1, 0:2]
  return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
  """Perform data augmentation by randomly dropping out strokes."""
  # drop each point within a line segments with a probability of prob
  # note that the logic in the loop prevents points at the ends to be dropped.
  result = []
  prev_stroke = [0, 0, 1]
  count = 0
  stroke = [0, 0, 1]  # Added to be safe.
  for i in range(len(strokes)):
    candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
    if candidate[2] == 1 or prev_stroke[2] == 1:
      count = 0
    else:
      count += 1
    urnd = np.random.rand()  # uniform random variable
    if candidate[2] == 0 and prev_stroke[2] == 0 and count > 3 and urnd < prob:
      stroke[0] += candidate[0]
      stroke[1] += candidate[1]
    else:
      stroke = candidate
      prev_stroke = stroke
      result.append(stroke)
  return np.array(result)

def random_scale_strokes(data, random_scale_factor=0.0):
  """Augment data by stretching x and y axis randomly [1-2*e, 1]."""
  x_scale_factor = (
      np.random.random() - 0.5) * 2 * random_scale_factor + 1.0 - random_scale_factor
  y_scale_factor = (
      np.random.random() - 0.5) * 2 * random_scale_factor + 1.0 - random_scale_factor
  result = np.copy(data)
  result[:, 0] *= x_scale_factor
  result[:, 1] *= y_scale_factor
  return result

# from https://stackoverflow.com/questions/44159861/how-do-i-parse-this-ndjson-file-in-python
@jit
def get_line(x1, y1, x2, y2):
  points = []
  issteep = abs(y2-y1) > abs(x2-x1)
  if issteep:
    x1, y1 = y1, x1
    x2, y2 = y2, x2
  rev = False
  if x1 > x2:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
    rev = True
  deltax = x2 - x1
  deltay = abs(y2-y1)
  error = int(deltax / 2)
  y = y1
  ystep = None
  if y1 < y2:
    ystep = 1
  else:
    ystep = -1
  for x in arange(x1, x2 + 1):
    if issteep:
      points.append((y, x))
    else:
      points.append((x, y))
    error -= deltay
    if error < 0:
      y += ystep
      error += deltax
  # Reverse the list if the coordinates were reversed
  if rev:
    points.reverse()
  return points   

@jit
def stroke_to_quickdraw(orig_data, max_dim_size=5.0):
  ''' convert back to list of points format, up to 255 dimensions '''
  data = np.copy(orig_data)
  data[:, 0:2] *= (255.0/max_dim_size) # to prevent overflow
  data = np.round(data).astype(np.int)
  line = []
  lines = []
  abs_x = 0
  abs_y = 0
  for i in arange(0, len(data)):
    dx = data[i,0]
    dy = data[i,1]
    abs_x += dx
    abs_y += dy
    abs_x = np.maximum(abs_x, 0)
    abs_x = np.minimum(abs_x, 255)
    abs_y = np.maximum(abs_y, 0)
    abs_y = np.minimum(abs_y, 255)  
    lift_pen = data[i, 2]
    line.append([abs_x, abs_y])
    if (lift_pen == 1):
      lines.append(line)
      line = []
  return lines

@jit
def create_image(stroke3, max_dim_size=5.0):
  image_dim = IMAGE_SIZE
  factor = 256/image_dim
  pixels = np.zeros((image_dim, image_dim))
  
  sketch = stroke_to_quickdraw(stroke3, max_dim_size=max_dim_size)

  x = -1
  y = -1

  for stroke in sketch:
    for i in arange(len(stroke)):
      if x != -1: 
        for point in get_line(stroke[i][0], stroke[i][1], x, y):
          pixels[int(point[0]/factor),int(point[1]/factor)] = 1
      pixels[int(stroke[i][0]/factor),int(stroke[i][1]/factor)] = 1
      x = stroke[i][0]
      y = stroke[i][1]
    x = -1
    y = -1
  return pixels.T.reshape(image_dim, image_dim, 1)

def package_augmentation(strokes,
                         random_drop_factor=0.15,
                         random_scale_factor=0.15,
                         max_dim_size=5.0):
  test_stroke = random_scale_strokes(
    augment_strokes(strokes, random_drop_factor),
    random_scale_factor)
  min_x, max_x, min_y, max_y = get_bounds(test_stroke, factor=1)
  rand_offset_x = (max_dim_size-max_x+min_x)*np.random.rand()
  rand_offset_y = (max_dim_size-max_y+min_y)*np.random.rand()
  test_stroke[0][0] += rand_offset_x
  test_stroke[0][1] += rand_offset_y
  return test_stroke

def scale_bound(stroke, average_dimension=10.0):
  """Scale an entire image to be less than a certain size."""
  # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
  # modifies stroke directly.
  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def clean_strokes(sample_strokes, factor=100):
  """Cut irrelevant end points, scale to pixel space and store as integer."""
  # Useful function for exporting data to .json format.
  copy_stroke = []
  added_final = False
  for j in range(len(sample_strokes)):
    finish_flag = int(sample_strokes[j][4])
    if finish_flag == 0:
      copy_stroke.append([
          int(round(sample_strokes[j][0] * factor)),
          int(round(sample_strokes[j][1] * factor)),
          int(sample_strokes[j][2]),
          int(sample_strokes[j][3]), finish_flag
      ])
    else:
      copy_stroke.append([0, 0, 0, 0, 1])
      added_final = True
      break
  if not added_final:
    copy_stroke.append([0, 0, 0, 0, 1])
  return copy_stroke


def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result


def get_max_len(strokes):
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes:
    ml = len(stroke)
    if ml > max_len:
      max_len = ml
  return max_len

def process_dataset(data_set, max_len, min_len):
  fixed_data = []
  all_length = []
  all_size = []
  for data in data_set:
    len_data = len(data)
    if len_data >= max_len or len_data < min_len:
      continue
    min_x, max_x, min_y, max_y = get_bounds(data)
    all_length.append(len(data))
    t = np.concatenate([[[-min_x, -min_y, 0]], data], axis=0).astype(np.float)
    factor = np.max([max_x-min_x, max_y-min_y])
    t[:, 0:2] /= factor
    fixed_data.append(t)
  return fixed_data

def get_dataset(class_name, max_len, min_len):
  print('loading', class_name)
  filename = os.path.join('npz', class_name+'.full.npz')
  load_data = np.load(filename, encoding='latin1')
  train_set_data = load_data['train']
  valid_set_data = load_data['valid']
  test_set_data = load_data['test']

  train_set_data = process_dataset(train_set_data, max_len, min_len)
  valid_set_data = process_dataset(valid_set_data, max_len, min_len)
  test_set_data = process_dataset(test_set_data, max_len, min_len)

  return train_set_data, valid_set_data, test_set_data

def get_test_dataset(class_name, max_len, min_len):
  print('loading', class_name)
  filename = os.path.join('npz', class_name+'.full.npz')
  load_data = np.load(filename, encoding='latin1')
  test_set_data = load_data['test']
  test_set_data = process_dataset(test_set_data, max_len, min_len)
  return test_set_data

def get_dataset_list(class_list, max_len, min_len):
  train = []
  valid = []
  test = []
  for c in class_list:
    train_set_data, valid_set_data, test_set_data = get_dataset(c, max_len, min_len)
    print("count (train/valid/test):",
          len(train_set_data),
          len(valid_set_data),
          len(test_set_data))
    train.append(train_set_data)
    valid.append(valid_set_data)
    test.append(test_set_data)
  return train, valid, test

def get_test_dataset_list(class_list, max_len, min_len):
  test = []
  for c in class_list:
    test_set_data = get_test_dataset(c, max_len, min_len)
    print("count (test):",
          len(test_set_data))
    test.append(test_set_data)
  return test

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

def translate_latent(old_kanji_z, translate_model):
  BATCH_SIZE = 1
  temperature = 0.25
  OUTWIDTH = Z_SIZE
  [logmix, mean, logstd] = translate_model.sess.run([translate_model.out_logmix, translate_model.out_mean, translate_model.out_logstd], {translate_model.x: old_kanji_z})
  # adjust temperatures
  logmix2 = np.copy(logmix)/temperature
  logmix2 = logmix2 - np.reshape(logmix2.max(axis=2), (BATCH_SIZE, Z_SIZE, 1))
  logmix2 = np.exp(logmix2)
  logmix2 /= logmix2.sum(axis=2).reshape(BATCH_SIZE, OUTWIDTH, 1)
  #mixture_idx = np.zeros((BATCH_SIZE, OUTWIDTH))
  chosen_mean = np.zeros((BATCH_SIZE, OUTWIDTH))
  #chosen_logstd = np.zeros((BATCH_SIZE, OUTWIDTH))
  for i in range(BATCH_SIZE):
    for j in range(OUTWIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[i][j])
      #mixture_idx[i][j] = idx
      chosen_mean[i][j] = mean[i][j][idx]
      #chosen_logstd[i][j] = logstd[i][j][idx]
  #rand_gaussian = np.random.randn(BATCH_SIZE, OUTWIDTH)*np.sqrt(temperature)
  #predict_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
  return chosen_mean

class OldKanjiLoader(object):
  """Class for loading data."""

  def __init__(self,
               stroke_data,
               stroke_label,
               unicode_stroke_label,
               old_kanji,
               new_kanji,
               unicode_label,
               vae_old,
               vae_new,
               batch_size=RNN_BATCH_SIZE,
               max_seq_length=134,
               scale_factor=1.6808838/10.,
               random_scale_factor=0.0,
               augment_stroke_prob=0.0,
               limit=1000):
    # process stroke_sets:
    strokes = process_dataset(stroke_data, max_seq_length, 1)
    self.stroke_unicode = stroke_label
    
    self.unicode_stroke_label = unicode_stroke_label # maybe not used
    self.old_kanji = old_kanji
    self.new_kanji = new_kanji
    self.unicode_old_label = unicode_label
    self.vae_old = vae_old
    self.vae_new = vae_new

    self.set_len = [len(strokes)]

    self.num_sets = 1
    self.set_ranges = np.concatenate([[0],np.cumsum(self.set_len)])
    self.batch_size = batch_size  # minibatch size
    self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
    self.max_dim_size = 1.0/scale_factor # maximum dimension of stroke image
    self.scale_factor = scale_factor  # divide offsets by this factor
    self.random_scale_factor = random_scale_factor  # data augmentation method
    # Removes large gaps in the data. x and y offsets are clamped to have
    # absolute value no greater than this limit.
    self.limit = limit
    self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
    self.start_stroke_token = [0, 0, 0, 1, 0]  # S_0 in sketch-rnn paper
    # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
    # sorted by size)
    self.process_images = True
    self.preprocess(strokes)

  def preprocess(self, strokes):
    """Remove entries from strokes having > max_seq_length points."""
    raw_data = []
    count_data = 0

    for i in range(len(strokes)):
      data = strokes[i]
      if len(data) <= (self.max_seq_length):
        count_data += 1
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data, dtype=np.float32)
        data[:, 0:2] /= self.scale_factor
        raw_data.append(data)
      else:
        assert False, "error: datapoint length >"+str(self.max_seq_length)
    self.strokes = raw_data
    print("total drawings <= max_seq_len is %d" % count_data)
    self.num_batches = int(count_data / self.batch_size)

  def random_sample(self):
    """Return a random sample, in stroke-3 format as used by draw_strokes."""
    sample = np.copy(random.choice(self.strokes))
    return sample

  def calculate_normalizing_scale_factor(self):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(self.strokes)):
      if len(self.strokes[i]) > self.max_seq_length:
        continue
      for j in range(len(self.strokes[i])):
        data.append(self.strokes[i][j, 0])
        data.append(self.strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

  def normalize(self, scale_factor=None):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    if scale_factor is None:
      scale_factor = self.calculate_normalizing_scale_factor()
    self.scale_factor = scale_factor
    for i in range(len(self.strokes)):
      self.strokes[i][:, 0:2] /= self.scale_factor

  def _get_batch_from_indices(self, indices, translate_model):
    """Given a list of indices, return the potentially augmented batch."""
    x_batch = []
    latents = []
    images = []
    seq_len = []
    batch_cat = []
    for idx in range(len(indices)):
      i = indices[idx]
      batch_cat.append(self.stroke_unicode[i])
      data_copy = package_augmentation(self.strokes[i],
                                       random_drop_factor=self.augment_stroke_prob,
                                       random_scale_factor=self.random_scale_factor,
                                       max_dim_size=self.max_dim_size)
      if self.process_images:
        unicode = self.stroke_unicode[i]
        if unicode in self.unicode_old_label:
          im = augment_pixel(random.choice(self.old_kanji[unicode])).astype(np.float32)/255.0
          images.append(im)
          z_old = self.vae_old.encode_mu(im.reshape(1, 64, 64, 1))
          latents.append(translate_latent(z_old, translate_model))
        else:
          im = augment_pixel(random.choice(self.new_kanji[unicode])).astype(np.float32)/255.0
          images.append(im)
          z_new = self.vae_new.encode(im.reshape(1, 64, 64, 1))
          latents.append(z_new)
      x_batch.append(data_copy)
      length = len(data_copy)
      seq_len.append(length)
    seq_len = np.array(seq_len, dtype=int)
    # We return four things: stroke-3 format, stroke-5 format, list of seq_len, pixel_images
    latents = np.concatenate(latents, axis=0)
    return [x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len, latents, images, batch_cat]

  def random_batch(self, translate_model):
    """Return a randomised portion of the training data."""
    # naive method just randomizes indices, i.e.:
    # idx = np.random.choice(np.arange(len(self.strokes)), self.batch_size)
    # to avoid unbalanced datasets, we randomize on categories first:
    idx = []
    for i in range(self.batch_size):
      category = 0
      idx_lo = self.set_ranges[category]
      idx_hi = self.set_ranges[category+1]
      sample_idx = np.random.randint(idx_lo, idx_hi)
      idx.append(sample_idx)
    return self._get_batch_from_indices(idx, translate_model)

  def get_batch(self, idx, translate_model):
    """Get the idx'th batch from the dataset."""
    assert idx >= 0, "idx must be non negative"
    assert idx < self.num_batches, "idx must be less than the number of batches"
    start_idx = idx * self.batch_size
    indices = range(start_idx, start_idx + self.batch_size)
    return self._get_batch_from_indices(indices, translate_model)

  def pad_batch(self, batch, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
    assert len(batch) == self.batch_size
    for i in range(self.batch_size):
      l = len(batch[i])
      assert l <= max_len
      result[i, 0:l, 0:2] = batch[i][:, 0:2]
      result[i, 0:l, 3] = batch[i][:, 2]
      result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
      result[i, l:, 4] = 1
      # put in the first token, as described in sketch-rnn methodology
      result[i, 1:, :] = result[i, :-1, :]
      result[i, 0, :] = 0
      result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      result[i, 0, 3] = self.start_stroke_token[3]
      result[i, 0, 4] = self.start_stroke_token[4]
    return result

def train_vae(unicode_label, old_train, new_kanji, batch_size=50, vae_name="vae_old", kuzushiji=True):
  reset_graph()
  vae = ConvVAE(z_size=64, batch_size=batch_size, learning_rate=0.001)

  for i in range(20000):
    # train loop

    batch_x, batch_y, batch_u = random_batch(batch_size, unicode_label, old_train, new_kanji)

    batch = batch_y
    if kuzushiji:
      batch = batch_x

    feed = {
      vae.x: batch,
      vae.y: batch
    }

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op], feed)

    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 2500 == 0):
      vae.save_model(vae_name)

def train_latent2latent(unicode_label, old_train, new_kanji):
  BATCH_SIZE = 3000
  Z_SIZE = 64

  reset_graph()
  model = Latent2Latent(batch_size=BATCH_SIZE, learning_rate=0.001)
  
  vae_old = ConvVAE(z_size=Z_SIZE, batch_size=BATCH_SIZE, gpu_mode=True, is_training=False, reuse=True)
  vae_old.load_checkpoint("vae_old")
  vae_new = ConvVAE(z_size=Z_SIZE, batch_size=BATCH_SIZE, gpu_mode=True, is_training=False, reuse=True)
  vae_new.load_checkpoint("vae_new")
  
  batch_x, batch_y, batch_u = random_batch(BATCH_SIZE, unicode_label, old_train, new_kanji)

  for i in range(30000):
    # train loop

    batch_x, batch_y, batch_u = random_batch(BATCH_SIZE, unicode_label, old_train, new_kanji)

    batch_old_z = vae_old.encode(batch_x)
    batch_new_z = vae_new.encode(batch_y)

    feed = {
      model.x: batch_old_z,
      model.y: batch_new_z
    }

    (train_loss, train_step, _) = model.sess.run([model.loss, model.global_step, model.train_op], feed)

    if ((train_step+1) % 50 == 0):
      print("step", (train_step+1), train_loss)
    if ((train_step+1) % 1000 == 0):
      model.save_model("image2image")

def dump_result(batch_num, kanji_set, new_kanji, translate_model, vae_old, vae_new, sample_model, result_dir="result", display_mode=False, num_trial=10):
  factor = 0.10
  batch_stroke3, batch_strokes, batch_seq_len, batch_latents, batch_images, batch_label = kanji_set.get_batch(batch_num, translate_model)
  idx = 0 # np.random.randint(0, len(batch_strokes))
  unicode_name = batch_label[idx]
  filename = hex(unicode_name)
  print("Unicode", chr(batch_label[idx]))
  print("Stroke (Answer)")
  draw_strokes(to_normal_strokes(batch_strokes[idx]), factor, svg_filename=os.path.join(result_dir, filename+"_gt.svg"), display_mode=display_mode)
  if display_mode:
    print("Kuzushiji")
    show_image(batch_images[idx])
    
  # VAE old
  old_z = vae_old.encode_mu(batch_images[0].reshape(1, 64, 64, 1))
  old_recon = vae_old.decode(old_z)
  if display_mode:
    print("Kuzushiji Reconstruction")
    show_image(old_recon)
  old_recon = np.array(old_recon*255).reshape(64,64).astype(np.uint8)
  imsave(os.path.join(result_dir, filename+"_vaeold.png"), 255-old_recon)
  
  # VAE new
  new_im = augment_pixel(random.choice(new_kanji[batch_label[0]])).astype(np.float32)/255.0
  new_z = vae_new.encode_mu(new_im.reshape(1, 64, 64, 1))
  new_recon = vae_new.decode(new_z)
  if display_mode:
    print("Modern Kanji")
    show_image(new_im)
    print("Modern Kanji Reconstruction")
    show_image(new_recon)
  new_recon = np.array(new_recon*255).reshape(64,64).astype(np.uint8)
  imsave(os.path.join(result_dir, filename+"_vaenew.png"), 255-new_recon)
  new_im = np.array(new_im*255).reshape(64,64).astype(np.uint8)
  imsave(os.path.join(result_dir, filename+"_target.png"), 255-new_im)

  orig = np.array(batch_images[idx]*255).astype(np.uint8)
  imsave(os.path.join(result_dir, filename+"_input.png"), 255-orig)
  raw_recon = vae_new.decode(batch_latents[idx].reshape(1, 64))
  if display_mode:
    print("Reconstruction of Predicted Latent Code")
    show_image(raw_recon)
  recon = np.array(raw_recon*255).reshape(64,64).astype(np.uint8)
  imsave(os.path.join(result_dir, filename+"_recon.png"), 255-recon)
  for i in range(num_trial):
    print("Sample Prediction #"+str(i)+":")
    strokes = sample(sample_model, batch_latents, temperature=0.15, gaussian_temperature=0.15)
    draw_strokes(to_normal_strokes(strokes), factor, svg_filename=os.path.join(result_dir, filename+"_"+str(i)+".svg"), display_mode=display_mode)

def load_kanji_rnn_test_data():
  unicode_label, test_unicode_label, old_train, old_test, new_kanji = load_data(sketch_rnn_mode=True)

  vae_old = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
  vae_old.load_checkpoint("vae_old")
  vae_new = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
  vae_new.load_checkpoint("vae_new")

  kanji_data = np.load('data/kanji_with_unicode.rdp25.npz', encoding='latin1')
  raw_stroke_data = kanji_data['kanji']
  raw_stroke_label = kanji_data['label']
  unicode_stroke_label = kanji_data['unique_label']

  stroke_data = []
  stroke_label = []
  for i in range(len(raw_stroke_data)):
    label = raw_stroke_label[i]
    if label in test_unicode_label:
      if label not in stroke_label:
        if len(old_train[label]) > 5:
          stroke_data.append(raw_stroke_data[i])
          stroke_label.append(label)

  kanji_set = OldKanjiLoader(
    stroke_data,
    stroke_label,
    unicode_stroke_label,
    old_test, new_kanji, test_unicode_label,
    vae_old, vae_new,
    batch_size=1,
    random_scale_factor=0.0,
    augment_stroke_prob=0.0)
  
  return kanji_set, new_kanji

def train_kanji_rnn():
  unicode_label, test_unicode_label, old_train, old_test, new_kanji = load_data(sketch_rnn_mode=True)
  reset_graph()
  translate_model = Latent2Latent(batch_size=1, gpu_mode=False, is_training=False)
  translate_model.load_checkpoint("image2image")

  vae_old = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
  vae_old.load_checkpoint("vae_old")
  vae_new = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
  vae_new.load_checkpoint("vae_new")

  kanji_data = np.load('data/kanji_with_unicode.rdp25.npz', encoding='latin1')
  raw_stroke_data = kanji_data['kanji']
  raw_stroke_label = kanji_data['label']
  unicode_stroke_label = kanji_data['unique_label']

  kanji_set = OldKanjiLoader(
    raw_stroke_data,
    raw_stroke_label,
    unicode_stroke_label,
    old_train, new_kanji, unicode_label,
    vae_old, vae_new,
    random_scale_factor=0.15,
    augment_stroke_prob=0.2)

  hps_train, hps_test, hps_sample = get_default_hparams()
  hps = hps_train
  model = MDNRNN(hps)
  
  with model.g.as_default():
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
      num_param = np.prod(var.get_shape().as_list())
      count_t_vars += num_param
      print(var.name, var.get_shape(), num_param)
    print("total trainable variables = %d" % (count_t_vars))

  # train loop
  start = time.time()
  step = 0
  avg_cost = []
  best_avg_cost = 1e9
  while (step < 1000000):

    step = model.sess.run(model.global_step)
    curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

    batch_stroke3, batch_strokes, batch_seq_len, batch_z, batch_images, batch_label = kanji_set.random_batch(translate_model)

    feed = {model.sequence: batch_strokes,
            model.batch_z: batch_z,
            model.lr: curr_learning_rate}

    (train_cost, state, train_step, _) = model.sess.run([model.cost, model.final_state, model.global_step, model.train_op], feed)
    avg_cost.append(train_cost)

    if (step%25==0 and step > 0):
      end = time.time()
      time_taken = end-start
      start = time.time()
      avg_cost_mean = np.mean(avg_cost)
      output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, avg_cost_mean, time_taken)
      print(output_log)
      assert np.isfinite(avg_cost_mean), "encountered NaN, leaving..."
    if (step % 250 == 0 and step > 0):
      if avg_cost_mean < best_avg_cost:
        best_avg_cost = avg_cost_mean
        print('saving')
        model.save_model("kanji_rnn", 0)
      avg_cost = []

def sample(s_model, z, seq_len=MAX_LEN, temperature=1.0, gaussian_temperature=1.0):
  """Samples a sequence from a pre-trained model."""
  
  hps = s_model.hps

  KMIX = hps.num_mixture # 5 mixtures
  WIDTH = hps.seq_width # 2 channels

  prev_stroke = np.zeros((1, 2, WIDTH+3))
  prev_stroke[0][0][WIDTH+1] = 1
  
  strokes = []
  strokes.append(prev_stroke[0, 0, :])

  batch_z = [z.flatten()]
  feed = {s_model.batch_z: batch_z}
  rnn_state = s_model.sess.run(s_model.initial_state, feed)
  
  for step in range(seq_len):

    feed = {s_model.sequence: prev_stroke,
            s_model.initial_state: rnn_state,
            s_model.batch_z : batch_z
            }

    [logmix, mean, logstd, logpen, next_state] = s_model.sess.run([s_model.out_logmix,
                                                                   s_model.out_mean,
                                                                   s_model.out_logstd,
                                                                   s_model.out_pen_logits,
                                                                   s_model.final_state],
                                                                   feed)

    # adjust temperatures for stroke
    logmix2 = np.copy(logmix)/temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(WIDTH, 1)

    # adjust temperatures for pen
    logpen2 = np.copy(logpen)/temperature
    logpen2 -= logpen2.max()
    logpen2 = np.exp(logpen2)
    logpen2 /= logpen2.sum(axis=1)
    next_pen_idx = get_pi_idx(np.random.rand(), logpen2[0])

    mixture_idx = np.zeros(WIDTH)
    chosen_mean = np.zeros(WIDTH)
    chosen_logstd = np.zeros(WIDTH)
    for j in range(WIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(WIDTH)*np.sqrt(gaussian_temperature)
    next_stroke = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    prev_stroke = np.zeros((1, 2, WIDTH+3))
    prev_stroke[0][0][:WIDTH] = next_stroke
    prev_stroke[0][0][WIDTH+next_pen_idx] = 1
  
    rnn_state = next_state
    strokes.append(prev_stroke[0, 0, :])
    
    if (next_pen_idx == 2):
      break
  return np.array(strokes)
