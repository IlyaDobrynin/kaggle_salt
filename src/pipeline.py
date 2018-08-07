import os
import numpy as np
import pandas as pd
from random import randint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.preprocessing.image import load_img

from tqdm import tqdm

from src.nn_models import unet_ding_han as udh
from src.nn_models.zf_turbo_unet.zf_turbo_unet import ZF_Unet

ROOT_DIR = r'C:\Programming\kaggle_salt'

img_size_ori = 101
img_size_target = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


def get_train_test():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, r"data/train.csv"), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(ROOT_DIR, r"data/depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = [
        np.array(load_img(os.path.join(ROOT_DIR, r"data/train/images/{}.png".format(idx)), grayscale=True)) / 255 for
        idx
        in tqdm(train_df.index)
    ]
    train_df["masks"] = [np.array(load_img(os.path.join(ROOT_DIR, r"data/train/masks/{}.png".format(idx)), grayscale=True)) / 255 for idx in
                         tqdm(train_df.index)]

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i

    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    return train_df, test_df


def show_images(in_df):
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    for i, idx in enumerate(in_df.index[:max_images]):
        img = in_df.loc[idx].images
        mask = in_df.loc[idx].masks
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.text(1, img_size_ori - 1, in_df.loc[idx].z, color="black")
        ax.text(img_size_ori - 1, 1, round(in_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
        ax.text(1, 1, in_df.loc[idx].coverage_class, color="black", ha="left", va="top")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")
    plt.show()


def training_proc(in_model, model_path, x_train, y_train, x_valid, y_valid):
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

    epochs = 200
    batch_size = 64

    history = in_model.fit(x_train, y_train,
                           validation_data=[x_valid, y_valid],
                           epochs=epochs,
                           verbose=2,
                           batch_size=batch_size,
                           callbacks=[early_stopping, model_checkpoint, reduce_lr],
                           shuffle=True,

                           )

    return history


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def validation_proc(model_path, train_df, x_valid, ids_valid):
    model = load_model(model_path)
    preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
    preds_valid = np.array([downsample(x) for x in preds_valid])
    y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    return threshold_best, iou_best


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def predict_submit(model_path, test_df, x_test, threshold_best, iou_best, out_path):
    model = load_model(model_path)
    preds_test = model.predict(x_test)
    pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in
                 enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(out_path, 'sub_{}.csv'.format(iou_best)))



def show_train_stats(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    plt.show()


if __name__ == "__main__":

    train_df, test_df = get_train_test()

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # sns.distplot(train_df.coverage, kde=False, ax=axs[0])
    # sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
    # plt.suptitle("Salt coverage")
    # axs[0].set_xlabel("Coverage")
    # axs[1].set_xlabel("Coverage class")
    # plt.show()

    # show_images(in_df=train_df)

    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    # model = udh.UNet((img_size_target, img_size_target, 1), start_ch=16, depth=5, batchnorm=True)
    model_class = ZF_Unet(img_rows=img_size_target,
                          img_cols=img_size_target,
                          img_channels=1)
    model = model_class.get_unet()
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.summary()

    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    model_path = os.path.join(ROOT_DIR, r'models/ding_han/keras.model')
    # hist = training_proc(model, model_path, x_train, y_train, x_valid, y_valid)
    # show_train_stats(hist)

    threshold_best, iou_best = validation_proc(model_path, train_df, x_valid, ids_valid)

    # print(threshold_best)
    x_test = np.array([upsample(np.array(load_img(os.path.join(ROOT_DIR, r'data/test/images/{}.png'.format(idx)),
                                                  grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1,
                                                                                                                   img_size_target,
                                                                                                                   img_size_target,
                                                                                                                   1)
    out_path = os.path.join(ROOT_DIR, r'sub/ding_han')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    predict_submit(model_path, test_df, x_test, threshold_best, iou_best, out_path)