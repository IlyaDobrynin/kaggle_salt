import os
import numpy as np
import pandas as pd

from random import randint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, model_from_json

from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.preprocessing.image import load_img

from tqdm import tqdm

from src.utils import images_proc as ip
from src.utils import losses
from src.pipe import train as tr
from src.pipe import validate as val
from src.pipe import submit as sb
from src.nn_models.zfturbo_unet import zf_unet_224_model as zfu
from src.nn_models.resnet_unet import u_net_model as rnu
from src.nn_models.resnet50_unet_selim import resnet50_unet as rfu
from src.nn_models.din_han_unet import unet_ding_han as udh

SEED = 42
ROOT_DIR = r'C:\Programming\kaggle_salt_challenge'

img_size_ori = 101
img_size_target = 224


def get_data(train_path, depths_path):
    train_df = pd.read_csv(os.path.join(ROOT_DIR, train_path), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(ROOT_DIR, depths_path), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    train_df["images"] = [np.array(load_img(os.path.join(ROOT_DIR, r'data/train/images/{}.png'.format(idx)), grayscale=True)) / 255 for idx
                          in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img(os.path.join(ROOT_DIR, r'data/train/masks/{}.png'.format(idx)), grayscale=True)) / 255 for idx in
                         tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i

    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    return train_df, depths_df, test_df


def train_proc(model, loss, metrics, model_name,
               train_images, train_masks, val_images, val_masks, batch_size):
    print('-' * 30 + 'TRAINING' + '-' * 30)
    train_generator, val_generator = tr.create_generator(train_images=train_images,
                                                         train_masks=train_masks,
                                                         val_images=val_images,
                                                         val_masks=val_masks,
                                                         batch_size=batch_size)

    model_json_name, model_h5_name = tr.train_model(model,
                                                    loss,
                                                    train_generator,
                                                    val_generator,
                                                    model_name,
                                                    metrics=metrics,
                                                    epochs=200,
                                                    patience=10,
                                                    optim_type='Adam',
                                                    learning_rate=0.001)

    return model_json_name, model_h5_name


def validation_proc(model_json_name, model_h5_name, train_df, x_valid, ids_valid):
    print('-' * 30 + 'VALIDATION' + '-' * 30)
    # Step 1: loading model from ../models
    print('-' * 30 + ' Loading model... ' + '-' * 30)
    json_file = open(os.path.join(ROOT_DIR, r'models', model_name, model_json_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(ROOT_DIR, r'models', model_name, model_h5_name))

    preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
    preds_valid = np.array([ip.downsample(x) for x in preds_valid])
    y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([val.iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    return threshold_best, iou_best


def predict_submit(model_json_name, model_h5_name, test_df, x_test, threshold_best, iou_best, out_path):
    print('-'*30 + 'PREDICT SUBMIT' + '-'*30)
    print('-' * 30 + ' Loading model... ' + '-' * 30)
    json_file = open(os.path.join(ROOT_DIR, r'models', model_name, model_json_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(ROOT_DIR, r'models', model_name, model_h5_name))

    preds_test = model.predict(x_test)
    pred_dict = {idx: sb.RLenc(np.round(ip.downsample(preds_test[i]) > threshold_best)) for i, idx in
                 enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(out_path, 'sub_{:.4f}.csv'.format(iou_best)))



if __name__ == "__main__":

    train_path = os.path.join(ROOT_DIR, r'data/train.csv')
    depths_path = os.path.join(ROOT_DIR, r'data/depths.csv')
    train_df, depths_df, test_df = get_data(train_path=train_path, depths_path=depths_path)

    print(np.array(train_df.images.map(ip.upsample).tolist()).shape)#.reshape(-1, img_size_target, img_size_target, 1))

    # ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    #     train_df.index.values,
    #     np.array(train_df.images.map(ip.upsample).tolist()),
    #     np.array(train_df.masks.map(ip.upsample).tolist()),
    #     train_df.coverage.values,
    #     train_df.z.values,
    #     test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)

    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(ip.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(ip.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)

    img_shape = (img_size_target, img_size_target, 1)
    # model = zfu.ZF_UNET_224(in_shape=img_shape)
    model = rnu.get_unet_resnet(img_shape)
    # model = udh.UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
    #      dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False)
    # model = rfu.resnet50_fpn((224, 224, 3))


    model_name = 'resnet_unet'
    batch_size = 16

    model_json_name, model_h5_name = train_proc(model=model,
                                                loss=losses.make_loss('weighted_bce_dice_loss'),
                                                metrics=['accuracy',
                                                         losses.dice_coef_border,
                                                         losses.dice_coef,
                                                         losses.binary_crossentropy,
                                                         losses.dice_coef_clipped],
                                                model_name=model_name,
                                                train_images=x_train, train_masks=y_train,
                                                val_images=x_valid, val_masks=y_valid,
                                                batch_size=batch_size)

    # model_json_name = r'C:\Programming\kaggle_salt_challenge\models\ding_han.json'
    # model_h5_name = r'C:\Programming\kaggle_salt_challenge\models\ding_han.h5'

    threshold_best, iou_best = validation_proc(model_json_name, model_h5_name, train_df, x_valid, ids_valid)

    print(threshold_best, iou_best)

    x_test = np.array([ip.upsample(np.array(load_img(os.path.join(ROOT_DIR, r'data/test/images/{}.png'.format(idx)),
                                                     grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1,
                                                                                          img_size_target,
                                                                                          img_size_target,
                                                                                          1)
    print(x_test.shape)
    out_path = os.path.join(ROOT_DIR, r'sub/{}'.format(model_name))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    predict_submit(model_json_name, model_h5_name, test_df, x_test, threshold_best, iou_best, out_path)