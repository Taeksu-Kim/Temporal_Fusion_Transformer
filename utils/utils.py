# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
  """Returns name of single column.
  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

  l = [tup[0] for tup in column_definition if tup[2] == input_type]

  if len(l) != 1:
    raise ValueError('Invalid number of columns for {}'.format(input_type))

  return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
  """Extracts the names of columns that correspond to a define data_type.
  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
  Returns:
    List of names for columns with data type specified.
  """
  return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
  ]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
  """Computes quantile loss for tensorflow.
  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Tensor for quantile loss.
  """

  # Checks quantile
  if quantile < 0 or quantile > 1:
    raise ValueError(
        'Illegal quantile value={}! Values should be between 0 and 1.'.format(
            quantile))

  prediction_underflow = y - y_pred
  q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
      1. - quantile) * tf.maximum(-prediction_underflow, 0.)

  return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
  """Computes normalised quantile loss for numpy arrays.
  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Float for normalised quantile loss.
  """
  prediction_underflow = y - y_pred
  weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

  quantile_loss = weighted_errors.mean()
  normaliser = y.abs().mean()

  return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.
  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
  """Creates tensorflow config for graphs to run on CPU or GPU.
  Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
  GPU machines.
  Args:
    tf_device: 'cpu' or 'gpu'
    gpu_id: GPU ID to use if relevant
  Returns:
    Tensorflow config.
  """

  if tf_device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
    tf_config = tf.ConfigProto(
        log_device_placement=False, device_count={'GPU': 0})

  else:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print('Selecting GPU ID={}'.format(gpu_id))

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

  return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
  """Saves Tensorflow graph to checkpoint.
  Saves all trainiable variables under a given variable scope to checkpoint.
  Args:
    tf_session: Session containing graph
    model_folder: Folder to save models
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope containing variables to save
  """
  # Save model
  if scope is None:
    saver = tf.train.Saver()
  else:
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

  save_path = saver.save(tf_session,
                         os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
  print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
  """Loads Tensorflow graph from checkpoint.
  Args:
    tf_session: Session to load graph into
    model_folder: Folder containing serialised model
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope to use.
    verbose: Whether to print additional debugging information.
  """
  # Load model proper
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print('Loading model from {0}'.format(load_path))

  print_weights_in_checkpoint(model_folder, cp_name)

  initial_vars = set(
      [v.name for v in tf.get_default_graph().as_graph_def().node])

  # Saver
  if scope is None:
    saver = tf.train.Saver()
  else:
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
  # Load
  saver.restore(tf_session, load_path)
  all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

  if verbose:
    print('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
    print('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
    print('All {0}'.format(','.join(all_vars)))

  print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
  """Prints all weights in Tensorflow checkpoint.
  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint
  Returns:
  """
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print_tensors_in_checkpoint_file(
      file_name=load_path,
      tensor_name='',
      all_tensors=True,
      all_tensor_names=True)

def cal_max_sample(data, InputTypes, config):
  id_col = get_single_col_by_input_type(InputTypes.ID, config.column_definition)
  time_col = get_single_col_by_input_type(InputTypes.TIME, config.column_definition)
  time_steps = config['total_time_steps']

  valid_sampling_locations = []
  for identifier, df in data.groupby(id_col):
    num_entries = len(df)
    if num_entries >= time_steps:
      valid_sampling_locations += [
          (identifier, time_steps + i)
          for i in range(num_entries - time_steps + 1)
      ]

  return len(valid_sampling_locations)

def batch_sampled_data(data, max_samples, InputTypes, config):
  """Samples segments into a compatible format.
  Args:
    data: Sources data to sample and batch
    max_samples: Maximum number of samples in batch
  Returns:
    Dictionary of batched data with the maximum samples specified.
  """
  time_steps = config['total_time_steps']
  input_size = config['input_size']
  output_size = config['output_size']
  num_encoder_steps = config['num_encoder_steps']
  
  if max_samples < 1:
    raise ValueError(
        'Illegal number of samples specified! samples={}'.format(max_samples))

  id_col = get_single_col_by_input_type(InputTypes.ID, config.column_definition)
  time_col = get_single_col_by_input_type(InputTypes.TIME, config.column_definition)

  data.sort_values(by=[id_col, time_col], inplace=True)

  print('Getting valid sampling locations.')
  valid_sampling_locations = []
  split_data_map = {}
  for identifier, df in data.groupby(id_col):
    print('Getting locations for {}'.format(identifier))
    num_entries = len(df)
    if num_entries >= time_steps:
      valid_sampling_locations += [
          (identifier, time_steps + i)
          for i in range(num_entries - time_steps + 1)
      ]
    split_data_map[identifier] = df

  inputs = np.zeros((max_samples, time_steps, input_size))
  outputs = np.zeros((max_samples, time_steps, output_size))
  time = np.empty((max_samples, time_steps, 1), dtype=object)
  identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

  if max_samples > 0 and len(valid_sampling_locations) > max_samples:
    print('Extracting {} samples...'.format(max_samples))
    ranges = [
        valid_sampling_locations[i] for i in np.random.choice(
            len(valid_sampling_locations), max_samples, replace=False)
    ]
  else:
    print('Max samples={} exceeds # available segments={}'.format(
        max_samples, len(valid_sampling_locations)))
    ranges = valid_sampling_locations

  id_col = get_single_col_by_input_type(InputTypes.ID, config.column_definition)
  time_col = get_single_col_by_input_type(InputTypes.TIME, config.column_definition)
  target_col = get_single_col_by_input_type(InputTypes.TARGET, config.column_definition)
  input_cols = [
      tup[0]
      for tup in config.column_definition
      if tup[2] not in {InputTypes.ID, InputTypes.TIME}
  ]

  for i, tup in enumerate(ranges):
    if (i + 1 % 1000) == 0:
      print(i + 1, 'of', max_samples, 'samples done...')
    identifier, start_idx = tup
    sliced = split_data_map[identifier].iloc[start_idx -
                                              time_steps:start_idx]
    inputs[i, :, :] = sliced[input_cols]
    outputs[i, :, :] = sliced[[target_col]]
    time[i, :, 0] = sliced[time_col]
    identifiers[i, :, 0] = sliced[id_col]

  sampled_data = {
      'inputs': inputs,
      'outputs': outputs[:, num_encoder_steps:, :],
      'active_entries': np.ones_like(outputs[:, num_encoder_steps:, :]),
      'time': time,
      'identifier': identifiers
  }

  return sampled_data

