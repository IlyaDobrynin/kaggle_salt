from skimage.transform import resize
import numpy as np
from skimage.morphology import *
from skimage import util
import cv2
from tqdm import tqdm
import os

img_size_t = 128

def image_pad(img, img_size_ori=101, img_size_target=img_size_t):
    if img_size_ori == img_size_target:
        return img
    pad_width = ((int(np.floor((img_size_t-img_size_ori)/2)),
                  int(np.ceil((img_size_t-img_size_ori)/2))),
                 (int(np.floor((img_size_t-img_size_ori)/2)),
                  int(np.ceil((img_size_t-img_size_ori)/2))))

    out_image = util.pad(img, pad_width, mode='reflect')
    # print(out_image.shape)
    return out_image


def image_crop(img, img_size_ori=101, img_size_target=img_size_t):
    if img_size_ori == img_size_target:
        return img
    crop_width = ((int(np.floor((img_size_t-img_size_ori)/2)),
                  int(np.ceil((img_size_t-img_size_ori)/2))),
                 (int(np.floor((img_size_t-img_size_ori)/2)),
                  int(np.ceil((img_size_t-img_size_ori)/2))))
    out_image = util.crop(img, crop_width)
    return out_image


def upsample(img, img_size_ori=101, img_size_target=img_size_t):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='symmetric', preserve_range=True)


def downsample(img, img_size_ori=101, img_size_target=img_size_t):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


def remove_small_instances(labels, max_hole_size=56, min_obj_size=50):
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.uint32)
    for i, label in tqdm(enumerate(labels)):
        remove_holes_img = remove_small_holes(label, min_size=max_hole_size)
        remove_objects_img = remove_small_objects(remove_holes_img, min_size=min_obj_size)
        out_labels[i, :, :] = remove_objects_img

    return out_labels


def morfling(labels):
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.uint32)
    for i, label in tqdm(enumerate(labels)):
        dilate_img = dilation(label)
        out_labels[i, :, :] = dilate_img
    np.save(file=os.path.join(r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\out_images\ding_han',
                              'morfled_preds.npy'),
            arr=out_labels)
    return out_labels


def bluring(labels):
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.float32)
    for i, label in tqdm(enumerate(labels)):
        # print(label.shape)
        blured_image = cv2.GaussianBlur(label, (5,5), 0)
        # print(blured_image.shape)
        out_labels[i, :, :] = blured_image
    np.save(file=os.path.join(r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\out_images\ding_han',
                              'blur_preds.npy'),
            arr=out_labels)
    return out_labels

if __name__ == '__main__':
    train_images = np.load(r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\in_images\train_images_down.npy')
    print(train_images.shape)
    out_labels = np.ndarray((train_images.shape[0], img_size_t, img_size_t), dtype=np.float32)
    for i, img in tqdm(enumerate(train_images)):
        out_labels[i, :, :] = image_pad(img)

    np.save(file=r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\in_images\train_pad.npy',
            arr=out_labels)