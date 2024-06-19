# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video friendly random augmentation.

This is a moddified copy of EfficientNet RandAugment here:
github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

"""
import inspect
import math
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf2
from tensorflow_addons import image as contrib_image
import tensorflow as tf

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf1.convert_to_tensor(image1)
    if factor == 1.0:
        return tf1.convert_to_tensor(image2)

    image1 = tf1.to_float(image1)
    image2 = tf1.to_float(image2)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf1.to_float(image1) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf1.cast(temp, tf1.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf1.cast(tf1.clip_by_value(temp, 0.0, 255.0), tf1.uint8)


def cutout(image, seed, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.
      seed: the random seed.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = tf1.shape(image)[0]
    image_width = tf1.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf2.random.stateless_uniform(
        seed=seed, shape=[], minval=0, maxval=image_height, dtype=tf1.int32
    )

    cutout_center_width = tf2.random.stateless_uniform(
        seed=seed, shape=[], minval=0, maxval=image_width, dtype=tf1.int32
    )

    lower_pad = tf1.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf1.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf1.maximum(0, cutout_center_width - pad_size)
    right_pad = tf1.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf1.pad(
        tf1.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf1.expand_dims(mask, -1)
    mask = tf1.tile(mask, [1, 1, 3])
    image = tf1.where(
        tf1.equal(mask, 0), tf1.ones_like(image, dtype=image.dtype) * replace, image
    )
    return image


def solarize(image, seed, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    del seed
    return tf1.where(image < threshold, image, 255 - image)


def solarize_add(image, seed, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    del seed
    added_image = tf1.cast(image, tf1.int64) + addition
    added_image = tf1.cast(tf1.clip_by_value(added_image, 0, 255), tf1.uint8)
    return tf1.where(image < threshold, added_image, image)


def color(image, seed, factor):
    """Equivalent of PIL Color."""
    del seed
    degenerate = tf1.image.grayscale_to_rgb(tf1.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image, seed, factor):
    """Equivalent of PIL Contrast."""
    del seed
    degenerate = tf1.image.rgb_to_grayscale(image)
    # Cast before calling tf1.histogram.
    degenerate = tf1.cast(degenerate, tf1.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf1.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf1.reduce_sum(tf1.cast(hist, tf1.float32)) / 256.0
    degenerate = tf1.ones_like(degenerate, dtype=tf1.float32) * mean
    degenerate = tf1.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf1.image.grayscale_to_rgb(tf1.cast(degenerate, tf1.uint8))
    return blend(degenerate, image, factor)


def brightness(image, seed, factor):
    """Equivalent of PIL Brightness."""
    del seed
    degenerate = tf1.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image, seed, bits):
    """Equivalent of PIL Posterize."""
    del seed
    shift = 8 - bits
    return tf1.bitwise.left_shift(tf1.bitwise.right_shift(image, shift), shift)


def rotate(image, seed, degrees, replace):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      seed: the random seed.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
    del seed
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = contrib_image.rotate(wrap(image), radians)
    return unwrap(image, replace)


def flip(image, seed, replace):
    del seed
    image = tf2.image.flip_left_right(wrap(image))
    return unwrap(image, replace)


def translate_x(image, seed, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    del seed
    image = contrib_image.translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image, seed, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    del seed
    image = contrib_image.translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def shear_x(image, seed, level, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    del seed
    image = contrib_image.transform(
        wrap(image), [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    return unwrap(image, replace)


def shear_y(image, seed, level, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    del seed
    image = contrib_image.transform(
        wrap(image), [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
    )
    return unwrap(image, replace)


def autocontrast(image, seed):
    """Implements Autocontrast function from PIL using TF ops.

    Args:
      image: A 3D uint8 tensor.
      seed: the random seed.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """

    del seed

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf1.to_float(tf1.reduce_min(image))
        hi = tf1.to_float(tf1.reduce_max(image))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf1.to_float(im) * scale + offset
            im = tf1.clip_by_value(im, 0.0, 255.0)
            return tf1.cast(im, tf1.uint8)

        result = tf1.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf1.stack([s1, s2, s3], 2)
    return image


def sharpness(image, seed, factor):
    """Implements Sharpness function from PIL using TF ops."""
    del seed
    orig_image = image
    image = tf1.cast(image, tf1.float32)
    # Make image 4D for conv operation.
    image = tf1.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = (
        tf1.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf1.float32, shape=[3, 3, 1, 1]
        )
        / 13.0
    )
    # Tile across channel dimension.
    kernel = tf1.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    with tf1.device("/cpu:0"):
        # Some augmentation that uses depth-wise conv will cause crashing when
        # training on GPU. See (b/156242594) for details.
        degenerate = tf1.nn.depthwise_conv2d(
            image, kernel, strides, padding="VALID", rate=[1, 1]
        )
    degenerate = tf1.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf1.squeeze(tf1.cast(degenerate, tf1.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf1.ones_like(degenerate)
    padded_mask = tf1.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf1.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf1.where(tf1.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image, seed):
    """Implements Equalize function from PIL using TF ops."""
    del seed

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf1.cast(im[:, :, c], tf1.int32)
        # Compute the histogram of the image channel.
        histo = tf1.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf1.where(tf1.not_equal(histo, 0))
        nonzero_histo = tf1.reshape(tf1.gather(histo, nonzero), [-1])
        step = (tf1.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf1.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf1.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf1.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf1.cond(
            tf1.equal(step, 0),
            lambda: im,
            lambda: tf1.gather(build_lut(histo, step), im),
        )

        return tf1.cast(result, tf1.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf1.stack([s1, s2, s3], 2)
    return image


def invert(image, seed):
    """Inverts the image pixels."""
    del seed
    image = tf1.convert_to_tensor(image)
    return 255 - image


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf1.shape(image)
    extended_channel = tf1.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf1.concat([image, extended_channel], 2)
    return extended


def unwrap(image, replace):
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
      image: A 3D Image Tensor with 4 channels.
      replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
      image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf1.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf1.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3]

    replace = tf1.concat([replace, tf1.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf1.where(
        tf1.equal(alpha_channel, 0),
        tf1.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image,
    )

    image = tf1.reshape(flattened_image, image_shape)
    image = tf1.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image


NAME_TO_FUNC = {
    "AutoContrast": autocontrast,
    "Equalize": equalize,
    "Flip": flip,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "Solarize": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "Contrast": contrast,
    "Brightness": brightness,
    "Sharpness": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
    "Cutout": cutout,
}


def _randomly_negate_tensor(tensor, seed):
    """With 50% prob turn the tensor negative."""
    rnd = tf2.random.stateless_uniform([], seed=seed)
    should_flip = tf1.cast(tf1.floor(rnd + 0.5), tf1.bool)
    final_tensor = tf1.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level, seed):
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate_tensor(level, seed)
    return (level,)


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0,)  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2.0 / (_MAX_LEVEL / level) + 0.9
    return (level,)


def _enhance_level_to_arg(level):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level, seed):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, seed)
    return (level,)


def _translate_level_to_arg(level, seed, translate_const):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, seed)
    return (level,)


def level_to_arg(hparams, seed):
    return {
        "AutoContrast": lambda level: (),
        "Equalize": lambda level: (),
        "Flip": lambda level: (),
        "Invert": lambda level: (),
        "Rotate": lambda level: _rotate_level_to_arg(level, seed),
        "Posterize": lambda level: (int((level / _MAX_LEVEL) * 4),),
        "Solarize": lambda level: (int((level / _MAX_LEVEL) * 256),),
        "SolarizeAdd": lambda level: (int((level / _MAX_LEVEL) * 110),),
        "Color": _enhance_level_to_arg,
        "Contrast": _enhance_level_to_arg,
        "Brightness": _enhance_level_to_arg,
        "Sharpness": _enhance_level_to_arg,
        "ShearX": lambda level: _shear_level_to_arg(level, seed),
        "ShearY": lambda level: _shear_level_to_arg(level, seed),
        # pylint:disable=g-long-lambda
        "Cutout": lambda level: (int((level / _MAX_LEVEL) * hparams["cutout_const"]),),
        "TranslateX": lambda level: _translate_level_to_arg(
            level, seed, hparams["translate_const"]
        ),
        "TranslateY": lambda level: _translate_level_to_arg(
            level, seed, hparams["translate_const"]
        ),
        # pylint:enable=g-long-lambda
    }


def _parse_policy_info(name, seed, prob, level, replace_value, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams, seed)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types
    if "prob" in inspect.getfullargspec(func)[0]:  # pylint:disable=deprecated-method
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    # pytype:disable=wrong-arg-types
    if "replace" in inspect.getfullargspec(func)[0]:  # pylint:disable=deprecated-method
        # Make sure replace is the final argument
        assert (
            "replace" == inspect.getfullargspec(func)[0][-1]
        )  # pylint:disable=deprecated-method
        args = tuple(list(args) + [replace_value])
    # pytype:enable=wrong-arg-types

    return (func, prob, args)


def randaugment(image, num_layers, magnitude, seeds):
    """Applies the RandAugment policy to `image`.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].
      seeds: The random seeds.

    Returns:
      The augmented version of `image`.
    """
    replace_value = [128] * 3
    tf1.logging.info("Using RandAug.")
    augmentation_hparams = {"cutout_const": 10, "translate_const": 10}
    available_ops = [
        "AutoContrast",
        "Equalize",
        "Flip",
        # 'Invert',
        "Rotate",
        "Posterize",
        # 'Solarize',
        "Color",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        # 'Cutout',
        # 'SolarizeAdd'
    ]

    for layer_num in range(num_layers):
        op_to_select = tf2.random.stateless_uniform(
            [], seed=seeds[0], maxval=len(available_ops), dtype=tf1.int32
        )
        random_magnitude = float(magnitude)
        with tf1.name_scope("randaug_layer_{}".format(layer_num)):
            for i, op_name in enumerate(available_ops):
                prob = tf2.random.stateless_uniform(
                    [], seed=seeds[1], minval=0.2, maxval=0.8, dtype=tf1.float32
                )
                func, _, args = _parse_policy_info(
                    op_name,
                    seeds[2],
                    prob,
                    random_magnitude,
                    replace_value,
                    augmentation_hparams,
                )
                image = tf1.cond(
                    tf1.equal(i, op_to_select),
                    # pylint:disable=g-long-lambda
                    lambda selected_func=func, selected_args=args: selected_func(
                        image, seeds[3], *selected_args
                    ),
                    # pylint:enable=g-long-lambda
                    lambda: image,
                )
    return image


def rand_aug(seeds, video, num_layers, magnitude):
    """RandAug for video with the same random seed for all frames."""
    image_aug = lambda a, x: randaugment(x, num_layers, magnitude, seeds)
    return tf.scan(image_aug, video)
