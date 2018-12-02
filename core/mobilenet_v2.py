# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Implementation of Mobilenet V2.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

import tensorflow as tf

import collections
import os
import contextlib

slim = tf.contrib.slim

def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size = tf.convert_to_tensor(
        [1, tf.shape(input_tensor)[1],
         tf.shape(input_tensor)[2], 1])
  else:
    kernel_size = [1, shape[1], shape[2], 1]
  output = pool_op(
      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
  # Recover output shape, for unknown shape.
  output.set_shape([None, 1, 1, None])
  return output

def _fixed_padding(inputs, kernel_size, rate=1):

  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                  [pad_beg[1], pad_end[1]], [0, 0]])
  return padded_inputs


def _split_divisible(num, num_ways, divisible_by=8):
  """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
  assert num % divisible_by == 0
  assert num / num_ways >= divisible_by
  # Note: want to round down, we adjust each split to match the total.
  base = num // num_ways // divisible_by * divisible_by
  result = []
  accumulated = 0
  for i in range(num_ways):
    r = base
    while accumulated + r < num * (i + 1) / num_ways:
      r += divisible_by
    result.append(r)
    accumulated += r
  assert accumulated == num
  return result

def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def expand_input_by_factor(n, divisible_by=8):
  return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)



def _split_divisible(num, num_ways, divisible_by=8):
  """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
  assert num % divisible_by == 0
  assert num / num_ways >= divisible_by
  # Note: want to round down, we adjust each split to match the total.
  base = num // num_ways // divisible_by * divisible_by
  result = []
  accumulated = 0
  for i in range(num_ways):
    r = base
    while accumulated + r < num * (i + 1) / num_ways:
      r += divisible_by
    result.append(r)
    accumulated += r
  assert accumulated == num
  return result

def split_conv(input_tensor,
               num_outputs,
               num_ways,
               scope,
               divisible_by=8,
               **kwargs):
  b = input_tensor.get_shape().as_list()[3]
  if num_ways == 1 or min(b // num_ways,
                          num_outputs // num_ways) < divisible_by:
    # Don't do any splitting if we end up with less than 8 filters
    # on either side.
    return slim.conv2d(input_tensor, num_outputs, [1, 1], scope=scope, **kwargs)

  outs = []
  input_splits = _split_divisible(b, num_ways, divisible_by=divisible_by)
  output_splits = _split_divisible(
      num_outputs, num_ways, divisible_by=divisible_by)
  inputs = tf.split(input_tensor, input_splits, axis=3, name='split_' + scope)
  base = scope
  for i, (input_tensor, out_size) in enumerate(zip(inputs, output_splits)):
    scope = base + '_part_%d' % (i,)
    n = slim.conv2d(input_tensor, out_size, [1, 1], scope=scope, **kwargs)
    n = tf.identity(n, scope + '_output')
    outs.append(n)
  return tf.concat(outs, 3, name=scope + '_concat')

@slim.add_arg_scope
def expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  project_activation_fn=tf.identity,
                  split_projection=1,
                  split_expansion=1,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  scope=None):
  with tf.variable_scope(scope, default_name='expanded_conv') as s, \
       tf.name_scope(s.original_name_scope):
    prev_depth = input_tensor.get_shape().as_list()[3]
    if  depthwise_location not in [None, 'input', 'output', 'expansion']:
      raise TypeError('%r is unknown value for depthwise_location' %
                      depthwise_location)

    depthwise_func = functools.partial(
        slim.separable_conv2d,
        num_outputs=None,
        kernel_size=kernel_size,
        depth_multiplier=depthwise_channel_multiplier,
        stride=stride,
        rate=rate,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope='depthwise')
    # b1 -> b2 * r -> b2
    #   i -> (o * r) (bottleneck) -> o
    input_tensor = tf.identity(input_tensor, 'input')
    net = input_tensor

    if callable(expansion_size):
      inner_size = expansion_size(num_inputs=prev_depth)
    else:
      inner_size = expansion_size

    if inner_size > net.shape[3]:
      net = split_conv(
          net,
          inner_size,
          num_ways=split_expansion,
          scope='expand',
          stride=1,
          normalizer_fn=normalizer_fn)

    if depthwise_location == 'expansion':
      net = depthwise_func(net)

    net = tf.identity(net, name='depthwise_output')
    if endpoints is not None:
      endpoints['depthwise_output'] = net

    # Note in contrast with expansion, we always have
    # projection to produce the desired output size.
    net = split_conv(
        net,
        num_outputs,
        num_ways=split_projection,
        stride=1,
        scope='project',
        normalizer_fn=normalizer_fn,
        activation_fn=project_activation_fn)
    if endpoints is not None:
      endpoints['projection_output'] = net

    if callable(residual):  # custom residual
      net = residual(input_tensor=input_tensor, output_tensor=net)
    elif (residual and
          # stride check enforces that we don't add residuals when spatial
          # dimensions are None
          stride == 1 and
          # Depth matches
          net.get_shape().as_list()[3] ==
          input_tensor.get_shape().as_list()[3]):
      net += input_tensor
    return tf.identity(net, name='output')


class NoOpScope(object):

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False

def safe_arg_scope(funcs, **kwargs):

  filtered_args = {name: value for name, value in kwargs.items()
                   if value is not None}
  if filtered_args:
    return slim.arg_scope(funcs, **filtered_args)
  else:
    return NoOpScope()


@contextlib.contextmanager
def _set_arg_scope_defaults(defaults):
    if hasattr(defaults, 'items'):
        items = list(defaults.items())
    else:
        items = defaults
    if not items:
        yield
    else:
        func, default_arg = items[0]
        with slim.arg_scope(func, **default_arg):
            with _set_arg_scope_defaults(items[1:]):
                yield

@contextlib.contextmanager
def _scope_all(scope, default_scope=None):
  with tf.variable_scope(scope, default_name=default_scope) as s,\
       tf.name_scope(s.original_name_scope):
    yield s

expand_input = expand_input_by_factor
@slim.add_arg_scope
def depth_multiplier_func(output_params,
                     multiplier,
                     divisible_by=8,
                     min_depth=8,
                     **unused_kwargs):
  if 'num_outputs' not in output_params:
    return
  d = output_params['num_outputs']
  output_params['num_outputs'] = _make_divisible(d * multiplier, divisible_by,
                                                 min_depth)

_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])

def op(opfunc, **params):
  multiplier = params.pop('multiplier_transorm', depth_multiplier_func)
  return _Op(opfunc, params=params, multiplier_func=multiplier)

V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(expanded_conv, stride=2, num_outputs=24),
        op(expanded_conv, stride=1, num_outputs=24),
        op(expanded_conv, stride=2, num_outputs=32),
        op(expanded_conv, stride=1, num_outputs=32),
        op(expanded_conv, stride=1, num_outputs=32),
        op(expanded_conv, stride=2, num_outputs=64),
        op(expanded_conv, stride=1, num_outputs=64),
        op(expanded_conv, stride=1, num_outputs=64),
        op(expanded_conv, stride=1, num_outputs=64),
        op(expanded_conv, stride=1, num_outputs=96),
        op(expanded_conv, stride=1, num_outputs=96),
        op(expanded_conv, stride=1, num_outputs=96),
        op(expanded_conv, stride=2, num_outputs=160),
        op(expanded_conv, stride=1, num_outputs=160),
        op(expanded_conv, stride=1, num_outputs=160),
        op(expanded_conv, stride=1, num_outputs=320),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)

@slim.add_arg_scope
def mobilenet_base(
    inputs,
    conv_defs,
    multiplier=1.0,
    final_endpoint=None,
    output_stride=None,
    use_explicit_padding=False,
    scope=None,
    is_training=False):

  if multiplier <= 0:
    raise ValueError('multiplier is not greater than zero.')
  # Set conv defs defaults and overrides.
  conv_defs_defaults = conv_defs.get('defaults', {})
  conv_defs_overrides = conv_defs.get('overrides', {})
  if use_explicit_padding:
    conv_defs_overrides = copy.deepcopy(conv_defs_overrides)
    conv_defs_overrides[
        (slim.conv2d, slim.separable_conv2d)] = {'padding': 'VALID'}

  if output_stride is not None:
    if output_stride == 0 or (output_stride > 1 and output_stride % 2):
      raise ValueError('Output stride must be None, 1 or a multiple of 2.')

  with _scope_all(scope, default_scope='Mobilenet'), \
      safe_arg_scope([slim.batch_norm], is_training=is_training), \
      _set_arg_scope_defaults(conv_defs_defaults), \
      _set_arg_scope_defaults(conv_defs_overrides):
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    # Insert default parameters before the base scope which includes
    # any custom overrides set in mobilenet.
    end_points = {}
    scopes = {}
    for i, opdef in enumerate(conv_defs['spec']):
      params = dict(opdef.params)
      opdef.multiplier_func(params, multiplier)
      stride = params.get('stride', 1)
      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= stride
      else:
        layer_stride = stride
        layer_rate = 1
        current_stride *= stride
      # Update params.
      params['stride'] = layer_stride
      # Only insert rate to params if rate > 1.
      if layer_rate > 1:
        params['rate'] = layer_rate
      # Set padding
      if use_explicit_padding:
        if 'kernel_size' in params:
          net = _fixed_padding(net, params['kernel_size'], layer_rate)
        else:
          params['use_explicit_padding'] = True

      end_point = 'layer_%d' % (i + 1)
      try:
        net = opdef.op(net, **params)
      except Exception:
        print('Failed to create op %i: %r params: %r' % (i, opdef, params))
        raise
      end_points[end_point] = net
      scope = os.path.dirname(net.name)
      scopes[scope] = end_point
      if final_endpoint is not None and end_point == final_endpoint:
        break

    # Add all tensors that end with 'output' to
    # endpoints
    for t in net.graph.get_operations():
      scope = os.path.dirname(t.name)
      bn = os.path.basename(t.name)
      if scope in scopes and t.name.endswith('output'):
        end_points[scopes[scope] + '/' + bn] = t.outputs[0]
    return net, end_points

@slim.add_arg_scope
def mobilenet(inputs,
              reuse=None,
              scope='Mobilenet',
              **mobilenet_args):

  #is_training = mobilenet_args.get('is_training', False)
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

  with tf.variable_scope(scope, 'Mobilenet', reuse=reuse) as scope:
    inputs = tf.identity(inputs, 'input')
    net, end_points = mobilenet_base(inputs, scope=scope, **mobilenet_args)
    return net, end_points


@slim.add_arg_scope
def mobilenet_v2(input_tensor,
              depth_multiplier=1.0,
              scope='MobilenetV2',
              conv_defs=None,
              finegrain_classification_mode=False,
              min_depth=None,
              divisible_by=None,
              activation_fn=None,
              **kwargs):

  if conv_defs is None:
    conv_defs = V2_DEF
  if 'multiplier' in kwargs:
    raise ValueError('mobilenetv2 doesn\'t support generic '
                     'multiplier parameter use "depth_multiplier" instead.')
  if finegrain_classification_mode:
    conv_defs = copy.deepcopy(conv_defs)
    if depth_multiplier < 1:
      conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
  if activation_fn:
    conv_defs = copy.deepcopy(conv_defs)
    defaults = conv_defs['defaults']
    conv_defaults = (
        defaults[(slim.conv2d, slim.fully_connected, slim.separable_conv2d)])
    conv_defaults['activation_fn'] = activation_fn

  depth_args = {}
  # NB: do not set depth_args unless they are provided to avoid overriding
  # whatever default depth_multiplier might have thanks to arg_scope.
  if min_depth is not None:
    depth_args['min_depth'] = min_depth
  if divisible_by is not None:
    depth_args['divisible_by'] = divisible_by

  with slim.arg_scope((depth_multiplier_func,), **depth_args):
      return mobilenet(
          input_tensor,
          conv_defs=conv_defs,
          scope=scope,
          multiplier=depth_multiplier,
          **kwargs)
  

def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):

  # Note: do not introduce parameters that would change the inference
  # model here (for example whether to use bias), modify conv_def instead.
  batch_norm_params = {
      'decay': bn_decay,
      'is_training': is_training
  }
  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      normalizer_fn=slim.batch_norm), \
      slim.arg_scope([mobilenet_base, mobilenet], is_training=is_training),\
      safe_arg_scope([slim.batch_norm], **batch_norm_params), \
      safe_arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
    return s

