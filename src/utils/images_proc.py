from skimage.transform import resize
import numpy as np


def upsample(img, img_size_ori=101, img_size_target=224):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='symmetric', preserve_range=True)


# def upsample(img, img_size_ori=101, img_size_target=224):
#     if img_size_ori == img_size_target:
#         return img
#     return np.stack((resize(img, (img_size_target, img_size_target), mode='symmetric', preserve_range=True),)*3, -1)


def downsample(img, img_size_ori=101, img_size_target=224):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]