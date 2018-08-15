import os
import numpy as np
import pandas as pd
import datetime
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
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
img_size_target = 128


def get_model(json_path, h5_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5_path)

    return model


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


# def get_train_val_split(train_df, size):
#     ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
#         train_df.index.values,
#         np.array(train_df.images.map(ip.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
#         np.array(train_df.masks.map(ip.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
#         train_df.coverage.values,
#         train_df.z.values,
#         test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)


def train_proc(model, loss, metrics, model_name,
               train_images, train_masks, val_images, val_masks, time, batch_size, out_path):
    print('-' * 30 + ' TRAINING ' + '-' * 30)
    print('-' * 30 + ' Make generators... ' + '-' * 30)
    train_generator, val_generator = tr.create_generator(train_images=train_images,
                                                         train_masks=train_masks,
                                                         val_images=val_images,
                                                         val_masks=val_masks,
                                                         batch_size=batch_size)
    print('-' * 30 + ' Start training... ' + '-' * 30)
    model_json_name, model_h5_name = tr.train_model(model=model,
                                                    loss=loss,
                                                    out_model_path=out_path,
                                                    train_generator=train_generator,
                                                    val_generator=val_generator,
                                                    model_name=model_name,
                                                    metrics=metrics,
                                                    epochs=200,
                                                    patience=10,
                                                    optim_type='Adam',
                                                    learning_rate=0.001)
    print('-' * 30 + ' TRAINING END ' + '-' * 30)

    model_json_path = os.path.join(ROOT_DIR, r'models/{}/{}/{}'.format(model_name, time, model_json_name))
    model_h5_path = os.path.join(ROOT_DIR, r'models/{}/{}/{}'.format(model_name, time, model_h5_name))

    return model_json_path, model_h5_path


def validation_proc(model_json_path, model_h5_path, train_df, x_valid, ids_valid):
    print('-' * 30 + ' VALIDATION ' + '-' * 30)
    # Step 1: loading model from ../models
    model = get_model(json_path=model_json_path, h5_path=model_h5_path)

    print('-' * 30 + ' Predict validation dataset ' + '-' * 30)
    preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)

    print('-' * 30 + ' Calc iou and threshold ' + '-' * 30)
    preds_valid = np.array([ip.image_crop(x) for x in preds_valid])
    y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([val.iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    print("Best threshold: " + threshold_best + "\nBest IoU: " + iou_best)

    print('-' * 30 + ' VALIDATION END ' + '-' * 30)

    return threshold_best, iou_best


def predict_masks(model_json_path, model_h5_path, x_test):
    print('-'*30 + ' PREDICT ' + '-'*30)
    model = get_model(json_path=model_json_path, h5_path=model_h5_path)

    print('-' * 30 + ' Predict test dataset ' + '-' * 30)
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)

    return preds_test


def postprocessing(mask_array, threshold_best):
    print('-' * 30 + ' POSTPROCESSING ' + '-' * 30)
    # masks_blured = ip.bluring(labels=mask_array)
    masks_removed = ip.remove_small_instances(labels=np.int32(mask_array > threshold_best))
    # masks_morfed = ip.morfling(labels=masks_removed_inst)

    return masks_removed


def submit(pred_masks, test_df, time, threshold_best, iou_best, out_path):
    print('-' * 30 + ' SUBMIT ' + '-' * 30)
    pred_dict = {idx: sb.RLenc(np.round(ip.image_crop(pred_masks[i]) > threshold_best)) for i, idx in
                 enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(out_path, '{}_sub_{:.4f}.csv'.format(time, iou_best)))

    print('-' * 30 + ' PREDICT SUBMIT END ' + '-' * 30)


if __name__ == "__main__":

    train_path = os.path.join(ROOT_DIR, r'data/train.csv')
    depths_path = os.path.join(ROOT_DIR, r'data/depths.csv')
    train_df, depths_df, test_df = get_data(train_path=train_path, depths_path=depths_path)


    # ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    #     train_df.index.values,
    #     np.array(train_df.images.map(ip.upsample).tolist()),
    #     np.array(train_df.masks.map(ip.upsample).tolist()),
    #     train_df.coverage.values,
    #     train_df.z.values,
    #     test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)

    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(ip.image_pad).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(ip.image_pad).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)

    img_shape = (img_size_target, img_size_target, 1)
    # model = zfu.ZF_UNET_224(in_shape=img_shape)
    model = rnu.get_unet_resnet(img_shape)
    # model = udh.UNet(img_shape, out_ch=1, start_ch=32, depth=4, inc_rate=2., activation='relu',
    #      dropout=0.3, batchnorm=True, maxpool=True, upconv=True, residual=False)
    # model = rfu.resnet50_fpn(img_shape)

    model_names = {
        'ding_han': 'ding_han',
        'zfturbo_unet': 'zfturbo_unet',
        'resnet_unet': 'resnet_unet',
    }
    parameters = {
        'model_name': model_names['resnet_unet'],
        'batch_size': 16,
        'loss': 'weighted_bce_dice_loss',
        'metrics': [
            'accuracy',
            losses.mean_iou,
        ],
        'time': "{:%Y%m%dT%H%M}".format(datetime.datetime.now())
    }

    """
                                    TRAINING PROCESS
    """
    out_model_path = os.path.join(ROOT_DIR, r'models/{}/{}'.format(parameters['model_name'], parameters['time']))
    if not os.path.exists(out_model_path):
        os.mkdir(out_model_path)

    pd.DataFrame.from_dict(data=parameters).to_csv(path_or_buf=os.path.join(out_model_path, 'parameters.csv'))

    model_json_path, model_h5_path = train_proc(model=model,
                                                loss=losses.make_loss(parameters['loss']),
                                                metrics=parameters['metrics'],
                                                model_name=parameters['model_name'],
                                                train_images=x_train, train_masks=y_train,
                                                val_images=x_valid, val_masks=y_valid,
                                                batch_size=parameters['batch_size'],
                                                time=parameters['time'],
                                                out_path=out_model_path)

    # model_json_path = r'C:\Programming\kaggle_salt_challenge\models\ding_han\20180813T1744\ding_han.json'
    # model_h5_path = r'C:\Programming\kaggle_salt_challenge\models\ding_han\20180813T1744\ding_han.h5'

    """
                                       VALIDATION PROCESS
    """
    threshold_best, iou_best = validation_proc(model_json_path=model_json_path,
                                               model_h5_path=model_h5_path,
                                               train_df=train_df,
                                               x_valid=x_valid,
                                               ids_valid=ids_valid)

    print(threshold_best, iou_best)

    """
                                       PREDICT PROCESS
    """
    x_test = np.array([ip.image_pad(np.array(load_img(os.path.join(ROOT_DIR, r'data/test/images/{}.png'.format(idx)),
                                                     grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1,
                                                                                          img_size_target,
                                                                                          img_size_target,
                                                                                          1)

    pred_masks = predict_masks(model_json_path=model_json_path,
                               model_h5_path=model_h5_path,
                               x_test=x_test)

    """
                                       POSTPROCESSING
    """

    pred_postproc_masks = postprocessing(mask_array=pred_masks, threshold_best=threshold_best)


    """
                                       SUBMIT PROCESS
    """
    out_path = os.path.join(ROOT_DIR, r'sub/{}'.format(parameters['model_name']))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    submit(pred_masks=pred_postproc_masks,
           test_df=test_df,
           time=parameters['time'],
           threshold_best=threshold_best,
           iou_best=iou_best,
           out_path=out_path)

