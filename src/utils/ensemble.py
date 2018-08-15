import  os
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

from sklearn.model_selection import train_test_split

from src.utils import images_proc as ip

from keras.models import model_from_json
from keras.preprocessing.image import load_img

from src.utils import images_proc as ip
from src.pipe import validate as val
from src import pipeline as pipe

import datetime

from tqdm import tqdm

ROOT_DIR = r'C:\Programming\kaggle_salt_challenge'


img_size = pipe.img_size_target
img_ori_size = pipe.img_size_ori


def get_model(json_path, h5_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5_path)

    return model


def pred_save_npy(models, data, weights, img_size, out_path, type):

    if type == 'val':
        size = img_ori_size
    else:
        size = img_size
    out_array = np.ndarray((len(models),
                            data.shape[0],
                            size,
                            size), dtype=np.float32)

    for i, k in enumerate(models.keys()):
        print("Predict {} set for {} model".format(type, k))
        model = get_model(json_path=models[k][0],
                          h5_path=models[k][1])
        preds = model.predict(data).reshape(-1, img_size, img_size)

        if type == 'val':
            preds = np.array([ip.image_crop(x) for x in preds])

        out_array[i, :, :, :] = preds

    ensemble_arr = np.ndarray((out_array.shape[1],
                               out_array.shape[2],
                               out_array.shape[3]), dtype=np.float32)
    # for i in range(out_array.shape[0]):
    #     ensemble_arr += out_array[i, :, :, :] * weights[i]
    ensemble_arr = gmean(out_array, axis=0)
    out_file = os.path.join(out_path, 'ensemble_{}.npy'.format(type))
    np.save(file=out_file, arr=ensemble_arr)

    print(ensemble_arr.shape)

    return out_file


def validate_ensemble(data_path, ids_valid):
    data = np.load(data_path)
    data_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([val.iou_metric_batch(data_ori, np.int32(data > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    # print("Best threshold: " + threshold_best + "\nBest IoU: " + iou_best)

    return threshold_best, iou_best


def predict_masks(model, x_test, img_size, model_name, out_file):
    preds_test = model.predict(x_test).reshape(-1, img_size, img_size)

    np.save(file=out_file, arr=preds_test)

    return preds_test


if __name__ == "__main__":

    print("LOAD DATA")
    train_path = os.path.join(ROOT_DIR, r'data/train.csv')
    depths_path = os.path.join(ROOT_DIR, r'data/depths.csv')
    train_df, depths_df, test_df = pipe.get_data(train_path=train_path, depths_path=depths_path)

    print("MAKE SPLIT")
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(ip.image_pad).tolist()).reshape(-1, img_size, img_size, 1),
        np.array(train_df.masks.map(ip.image_pad).tolist()).reshape(-1, img_size, img_size, 1),
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=pipe.SEED)

    model_names = ['ding_han_1',
                   'ding_han_2',
                   'zfturbo_unet']

    ensemble_parameters = {
        'models': {
        'ding_han_1': [os.path.join(ROOT_DIR, r'models/ding_han/20180812T0113_LB0.790/ding_han.json'),
                       os.path.join(ROOT_DIR, r'models/ding_han/20180812T0113_LB0.790/ding_han.h5'),
                       0.790],
        # 'ding_han_2': [os.path.join(ROOT_DIR, r'models/ding_han/20180814T1104_LB0.783/ding_han.json'),
        #                os.path.join(ROOT_DIR, r'models/ding_han/20180814T1104_LB0.783/ding_han.h5'),
        #                0.783],
        'zfturbo_unet': [os.path.join(ROOT_DIR, r'models/zfturbo_unet/20180814T1501_LB0.809/zfturbo_unet.json'),
                         os.path.join(ROOT_DIR, r'models/zfturbo_unet/20180814T1501_LB0.809/zfturbo_unet.h5'),
                         0.809]
        },
        'time': '{:%Y%m%dT%H%M}'.format(datetime.datetime.now()),
        'weights': np.asarray([0.5, 0.5], dtype=np.float32)
    }

    out_path = os.path.join(r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\ensemble',
                            ensemble_parameters['time'])
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    type = 'val'
    pd.DataFrame.from_dict(data=ensemble_parameters['models']).to_csv(path_or_buf=os.path.join(out_path, 'model_parameters.csv'))
    np.savetxt(os.path.join(out_path, 'weights.csv'), ensemble_parameters['weights'], delimiter=';')

    print("MAKE VALID ENSEMBLE")
    val_npy_path = pred_save_npy(models=ensemble_parameters['models'],
                  data=x_valid,
                  weights=ensemble_parameters['weights'],
                  img_size=img_size,
                  out_path=out_path,
                  type=type)
    # data_path = r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\ensemble\20180815T1514\ensemble_val.npy'

    threshold_best, iou_best = validate_ensemble(data_path=val_npy_path,
                                                 ids_valid=ids_valid)
    print(threshold_best, iou_best)

    print("MAKE TEST ENSEMBLE")
    x_test = np.array([ip.image_pad(np.array(load_img(os.path.join(ROOT_DIR, r'data/test/images/{}.png'.format(idx)),
                                                      grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size, img_size, 1)
    type = 'test'
    test_npy_path = pred_save_npy(models=ensemble_parameters['models'],
                                  data=x_test,
                                  weights=ensemble_parameters['weights'],
                                  img_size=img_size,
                                  out_path=out_path,
                                  type=type)

    pred_data = np.load(test_npy_path)

    pred_postproc_masks = pipe.postprocessing(mask_array=pred_data, threshold_best=threshold_best)

    sub_path = os.path.join(ROOT_DIR, r'sub/ensemble/{}'.format(ensemble_parameters['time']))
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    print("CREATE SUBMIT")
    pipe.submit(pred_masks=pred_postproc_masks,
                test_df=test_df,
                time=ensemble_parameters['time'],
                threshold_best=threshold_best,
                iou_best=iou_best,
                out_path=sub_path)





