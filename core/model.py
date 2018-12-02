# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
import tensorflow as tf
from core import dense_prediction_cell
from core import utils
from core import preprocess_utils
import functools
from core import mobilenet_v2
from core import xception

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

_scale_dimension = utils.scale_dimension
_split_separable_conv2d = utils.split_separable_conv2d
_preprocess_zero_mean_unit_range = preprocess_utils.preprocess_zero_mean_unit_range
_preprocess_subtract_imagenet_mean = preprocess_utils.preprocess_subtract_imagenet_mean
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'

def _mobilenet_v2(net,
                  depth_multiplier,
                  output_stride,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
  with tf.variable_scope(scope, 'MobilenetV2', [net], reuse=reuse) as scope:
    return mobilenet_v2.mobilenet_v2(
        net,
        conv_defs=mobilenet_v2.V2_DEF,
        depth_multiplier=depth_multiplier,
        min_depth=8 if depth_multiplier == 1.0 else 1,
        divisible_by=8 if depth_multiplier == 1.0 else 1,
        final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT,
        output_stride=output_stride,
        scope=scope)

# Names for end point features used for encoder_out_stride
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features.
networks_to_feature_maps = {
    'mobilenet_v2': {
        DECODER_END_POINTS: ['layer_4/depthwise_output'],
    },
    'xception_41': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
    'xception_65': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
    'xception_71': {
        DECODER_END_POINTS: [
            'entry_flow/block3/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
}

name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'xception_41': 'xception_41',
    'xception_65': 'xception_65',
    'xception_71': 'xception_71',
}

networks_map = {
    'mobilenet_v2': _mobilenet_v2,
    'xception_41': xception.xception_41,
    'xception_65': xception.xception_65,
    'xception_71': xception.xception_71,
}
# A map from network name to network arg scope.
arg_scopes_map = {
    'mobilenet_v2': mobilenet_v2.training_scope,
    'xception_41': xception.xception_arg_scope,
    'xception_65': xception.xception_arg_scope,
    'xception_71': xception.xception_arg_scope,
}

_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
}


def get_network(network_name, preprocess_images, arg_scope=None):
    if network_name not in networks_map:
        raise ValueError('Unsupported network %s.' % network_name)

    #arg_scope = arg_scope or mobilenet_v2.training_scope()
    arg_scope = arg_scope or arg_scopes_map[network_name]
    def _identity_function(inputs):
        return inputs
    if preprocess_images:
        preprocess_function = _PREPROCESS_FN[network_name]
        #preprocess_function = _preprocess_zero_mean_unit_range
    else:
        preprocess_function = _identity_function
    #func = _mobilenet_v2
    func = networks_map[network_name]
    @functools.wraps(func)
    def network_fn(inputs, *args, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(preprocess_function(inputs), *args, **kwargs)
    return network_fn


def extract_features_base(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     num_classes=None,
                     global_pool=False):

    if 'xception' in model_variant:
        arg_scope = arg_scopes_map[model_variant](
            weight_decay=weight_decay,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=1e-3,
            batch_norm_scale=True,
            regularize_depthwise=regularize_depthwise)
        features, end_points = get_network(
            model_variant, preprocess_images, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=name_scope[model_variant])
    elif 'mobilenet' in model_variant:
        arg_scope = arg_scopes_map[model_variant](
            is_training=(is_training and fine_tune_batch_norm),
            weight_decay=weight_decay)
        features, end_points = get_network(
            model_variant, preprocess_images, arg_scope)(
            inputs=images,
            depth_multiplier=depth_multiplier,
            output_stride=output_stride,
            reuse=reuse,
            scope=name_scope[model_variant],
            final_endpoint=final_endpoint)
    else:
        raise ValueError('Unknown model variant %s.' % model_variant)
    return features, end_points


def get_extra_layer_scopes(last_layers_contain_logits_only=False):

  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
        META_ARCHITECTURE_SCOPE,
    ]


def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):

  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_logits = multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)

  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):

  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = tf.image.resize_bilinear(
        scales_to_logits[MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = tf.argmax(logits, 3)

  return predictions


def _resize_bilinear(images, size, output_dtype=tf.float32):

  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):


  # Setup default values.
  if not image_pyramid:
      image_pyramid = [1.0]
  crop_height = (model_options.crop_size[0]
                 if model_options.crop_size else tf.shape(images)[1])
  crop_width = (model_options.crop_size[1]
               if model_options.crop_size else tf.shape(images)[2])

  # Compute the height, width for the output logits.
  logits_output_stride = (
      model_options.decoder_output_stride or model_options.output_stride)
  logits_height = _scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = _scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }
  for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = _scale_dimension(crop_height, image_scale)#513*2
      scaled_width = _scale_dimension(crop_width, image_scale)#513*2
      scaled_crop_size = [scaled_height, scaled_width] #[1026,1026]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)#[None,None]===>[1026,1026]
      if model_options.crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])#[1026, 1026]
    else:
      scaled_crop_size = model_options.crop_size
      scaled_images = images

    updated_options = model_options._replace(crop_size=scaled_crop_size)
    outputs_to_logits = _get_logits(
        scaled_images,
        updated_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    # Resize the logits to have the same dimension before merging.
    for output in sorted(outputs_to_logits):
      outputs_to_logits[output] = tf.image.resize_bilinear(
          outputs_to_logits[output], [logits_height, logits_width],
          align_corners=True)

    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(model_options.outputs_to_num_classes):
    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        tf.expand_dims(logits, axis=4)
        for logits in outputs_to_scales_to_logits[output].values()
    ]
    all_logits = tf.concat(all_logits, 4)
    merge_fn = (
        tf.reduce_max
        if model_options.merge_method == 'max' else tf.reduce_mean)
    outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=4)

  return outputs_to_scales_to_logits





def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):

  features, end_points = extract_features_base(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    if model_options.dense_prediction_cell_config is not None:
      tf.logging.info('Using dense prediction cell config.')
      dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
          config=model_options.dense_prediction_cell_config,
          hparams={
            'conv_rate_multiplier': 16 // model_options.output_stride,
          })
      concat_logits = dense_prediction_layer.build_cell(
          features,
          output_stride=model_options.output_stride,
          crop_size=model_options.crop_size,
          image_pooling_crop_size=model_options.image_pooling_crop_size,
          weight_decay=weight_decay,
          reuse=reuse,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm)
      return concat_logits, end_points
    else:
      # The following codes employ the DeepLabv3 ASPP module
      batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
      }

      with slim.arg_scope(
          [slim.conv2d, slim.separable_conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          padding='SAME',
          stride=1,
          reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          depth = 256
          branch_logits = []

          if model_options.add_image_level_feature:
            if model_options.crop_size is not None:
              image_pooling_crop_size = model_options.image_pooling_crop_size
              if image_pooling_crop_size is None:
                image_pooling_crop_size = model_options.crop_size
              pool_height = _scale_dimension(
                  image_pooling_crop_size[0],
                  1. / model_options.output_stride) #[513, 513] ===>[513/16, 513/16]
              pool_width = _scale_dimension(
                  image_pooling_crop_size[1],
                  1. / model_options.output_stride)
              image_feature = slim.avg_pool2d(
                  features, [pool_height, pool_width], [1, 1], padding='VALID')#[33,33]==>[1,1]
              resize_height = _scale_dimension(
                  model_options.crop_size[0],
                  1. / model_options.output_stride)
              resize_width = _scale_dimension(
                  model_options.crop_size[1],
                  1. / model_options.output_stride) #[1,1] ===> [33,33]
            else:
              # If crop_size is None, we simply do global pooling.
              pool_height = tf.shape(features)[1]
              pool_width = tf.shape(features)[2]
              image_feature = tf.reduce_mean(
                  features, axis=[1, 2], keepdims=True)
              resize_height = pool_height
              resize_width = pool_width
            image_feature = slim.conv2d(
                image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
            image_feature = _resize_bilinear(
                image_feature,
                [resize_height, resize_width],
                image_feature.dtype)
            # Set shape for resize_height/resize_width if they are not Tensor.
            if isinstance(resize_height, tf.Tensor):
              resize_height = None
            if isinstance(resize_width, tf.Tensor):
              resize_width = None
            image_feature.set_shape([None, resize_height, resize_width, depth])
            branch_logits.append(image_feature)

          # Employ a 1x1 convolution.
          branch_logits.append(slim.conv2d(features, depth, 1,
                                           scope=ASPP_SCOPE + str(0)))

          if model_options.atrous_rates:
            # Employ 3x3 convolutions with different atrous rates.
            for i, rate in enumerate(model_options.atrous_rates, 1):
              scope = ASPP_SCOPE + str(i)
              if model_options.aspp_with_separable_conv:
                aspp_features = _split_separable_conv2d(
                    features,
                    filters=depth,
                    rate=rate,
                    weight_decay=weight_decay,
                    scope=scope)
              else:
                aspp_features = slim.conv2d(
                    features, depth, 3, rate=rate, scope=scope)
              branch_logits.append(aspp_features)

          # Merge branch logits.
          concat_logits = tf.concat(branch_logits, 3)
          concat_logits = slim.conv2d(concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
          concat_logits = slim.dropout(
              concat_logits,
              keep_prob=0.9,
              is_training=is_training,
              scope=CONCAT_PROJECTION_SCOPE + '_dropout')
  return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):

  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  #mobilev2 has no part
  if model_options.decoder_output_stride is not None:
    if model_options.crop_size is None:
      height = tf.shape(images)[1]
      width = tf.shape(images)[2]
    else:
      height, width = model_options.crop_size
    decoder_height = _scale_dimension(height,
                                     1.0 / model_options.decoder_output_stride)
    decoder_width = _scale_dimension(width,
                                    1.0 / model_options.decoder_output_stride)
    features = refine_by_decoder(
        features,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_classes):
    outputs_to_logits[output] = get_branch_logits(
        features,
        model_options.outputs_to_num_classes[output],
        model_options.atrous_rates,
        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
        kernel_size=model_options.logits_kernel_size,
        weight_decay=weight_decay,
        reuse=reuse,
        scope_suffix=output)

  return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
  """Adds the decoder to obtain sharper segmentation results."""

  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
        feature_list = networks_to_feature_maps[
            model_variant][DECODER_END_POINTS]
        if feature_list is None:
          tf.logging.info('Not found any decoder end points.')
          return features
        else:
          decoder_features = features
          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              scope_name = name_scope[model_variant]
              #feature_name = '{}/{}'.format(
                  #'MobilenetV2', name)
              feature_name = '{}/{}'.format(
                  scope_name, name)
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
              h = (None if isinstance(decoder_height, tf.Tensor)
                   else decoder_height)
              w = (None if isinstance(decoder_width, tf.Tensor)
                   else decoder_width)
              decoder_features_list[j].set_shape([None, h, w, None])
            decoder_depth = 256
            if decoder_use_separable_conv:
              decoder_features = _split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = _split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')
            else:
              num_convs = 2
              decoder_features = slim.repeat(
                  tf.concat(decoder_features_list, 3),
                  num_convs,
                  slim.conv2d,
                  decoder_depth,
                  3,
                  scope='decoder_conv' + str(i))
          return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):

  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    #set the param by hand
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)
