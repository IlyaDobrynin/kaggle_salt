import keras.backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np

from src.pipe import validate


"""
                                UTILS
"""
def get_border_mask(pool_size, y_true):
    negative = 1 - y_true
    positive = y_true
    positive = K.pool2d(positive, pool_size=pool_size, padding="same")
    negative = K.pool2d(negative, pool_size=pool_size, padding="same")
    border = positive * negative
    return border


"""
                                IOU
"""


def IoU(y_true, y_pred, eps=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)


def zero_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)


def agg_loss(in_gt, in_pred):
    return - zero_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def my_iou_metric(y_true, y_pred):
    metric_value = tf.py_func(validate.iou_metric_batch, [y_true, y_pred], tf.float64)
    return metric_value


"""
                                DICE
"""
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)
    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
    y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))
    return dice_coef(y_true_f, y_pred_f)


def dice_coef_loss_border(y_true, y_pred):
    return (1 - dice_coef_border(y_true, y_pred)) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)


def dice_coef_clipped(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def bce_dice_loss_border(y_true, y_pred):
    return bce_border(y_true, y_pred) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + K.log(dice_coef(y_true, y_pred))


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_coef_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_coef_loss(y_true, y_pred)
    return loss


"""
                                JACARD
"""


def jacard_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_jacard_loss(y_true, y_pred):
    return bce_logdice_loss(y_true, y_pred) + jacard_coef_loss(y_true, y_pred)


def weighted_bce_dice_jacard_loss(y_true, y_pred):
    return weighted_bce_dice_loss(y_true, y_pred) + jacard_coef_loss(y_true, y_pred)


def weighted_bce_jacard_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + jacard_coef_loss(y_true, y_pred)
    return loss


"""
                                CROSSENTROPY
"""
def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))
    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
            K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))


def online_bootstrapping(y_true, y_pred, pixels=512, threshold=0.5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    difference = K.abs(y_true - y_pred)
    values, indices = K.tf.nn.top_k(difference, sorted=True, k=pixels)
    min_difference = (1 - threshold)
    y_true = K.tf.gather(K.gather(y_true, indices), K.tf.where(values > min_difference))
    y_pred = K.tf.gather(K.gather(y_pred, indices), K.tf.where(values > min_difference))
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def bce_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)

    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.tf.gather(y_true_f, K.tf.where(border > 0.5))
    y_pred_f = K.tf.gather(y_pred_f, K.tf.where(border > 0.5))

    return binary_crossentropy(y_true_f, y_pred_f)


def make_loss(loss_name):
    if loss_name == 'crossentropy':
        return K.binary_crossentropy
    elif loss_name == 'crossentropy_boot':
        def loss(y, p):
            return bootstrapped_crossentropy(y, p, 'hard', 0.9)
        return loss
    elif loss_name == 'dice':
        return dice_coef_loss
    elif loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)
        return loss
    elif loss_name == 'boot_soft':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=0.95)
        return loss
    elif loss_name == 'boot_hard':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='hard', alpha=0.95)
        return loss
    elif loss_name == 'online_bootstrapping':
        def loss(y, p):
            return online_bootstrapping(y, p, pixels=512 * 64, threshold=0.7)
        return loss
    elif loss_name == 'dice_coef_loss_border':
        return dice_coef_loss_border
    elif loss_name == 'bce_dice_loss_border':
        return bce_dice_loss_border
    elif loss_name == 'bce_logdice_loss':
        return bce_logdice_loss
    elif loss_name == 'weighted_bce_dice_loss':
        return weighted_bce_dice_loss
    elif loss_name == 'agg_loss':
        return agg_loss
    elif loss_name == 'dice_jacard_loss':
        return dice_jacard_loss
    elif loss_name == 'bce_dice_loss':
        return bce_dice_loss
    elif loss_name == 'dice_coef_loss_bce':
        return dice_coef_loss_bce
    elif loss_name == 'jacard_coef_loss':
        return jacard_coef_loss
    elif loss_name == 'weighted_bce_dice_jacard_loss':
        return weighted_bce_dice_jacard_loss
    elif loss_name == 'weighted_bce_jacard_loss':
        return weighted_bce_jacard_loss
    else:
        ValueError("Unknown loss.")








