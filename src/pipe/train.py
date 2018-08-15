import os
import pandas as pd
import gc
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image
import keras.backend as K

ROOT_DIR = r'C:\Programming\kaggle_salt_challenge'
SEED = 42


def create_generator(train_images, train_masks, val_images, val_masks, batch_size):

    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(shear_range=0.2,
                                             rotation_range=10,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             fill_mode='reflect')

    mask_datagen = image.ImageDataGenerator(shear_range=0.2,
                                            rotation_range=10,
                                            zoom_range=0.2,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together

    print(train_images.shape)

    image_datagen.fit(train_images, augment=True, seed=SEED)
    mask_datagen.fit(train_masks, augment=True, seed=SEED)

    x = image_datagen.flow(train_images, batch_size=batch_size, shuffle=True, seed=SEED)
    y = mask_datagen.flow(train_masks, batch_size=batch_size, shuffle=True, seed=SEED)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(val_images, augment=True, seed=SEED)
    mask_datagen_val.fit(val_masks, augment=True, seed=SEED)

    x_val = image_datagen_val.flow(val_images, batch_size=batch_size, shuffle=True, seed=SEED)
    y_val = mask_datagen_val.flow(val_masks, batch_size=batch_size, shuffle=True, seed=SEED)

    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)

    return train_generator, val_generator


def train_model(model, loss,
                train_generator, val_generator,
                model_name, out_model_path,
                metrics=None,
                epochs=200, patience=10, optim_type='Adam', learning_rate=0.001):

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
        ModelCheckpoint(os.path.join(out_model_path, '{}.h5'.format(model_name)),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        steps_per_epoch=200,
        validation_data=val_generator,
        validation_steps=200,
        verbose=2,
        # workers=4,
        # use_multiprocessing=True,
        callbacks=callbacks)

    model_h5_name = "{}.h5".format(model_name)
    model_json_name = save_model(model=model, model_name=model_name, model_save_path=out_model_path)
    # model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv(os.path.join(out_model_path, r'logs.csv'), index=False)
    del model
    K.clear_session()
    gc.collect()
    print('Training is finished...')

    return model_json_name, model_h5_name


def save_model(model, model_name, model_save_path):
    """
    Save trained model and weights
    :param model: Fitted model
    :param model_name: Name of the model to save
    :return: Nothing to return
    """

    model_json_name = "{}.json".format(model_name)
    # model_h5_name = "{}.h5".format(model_name)

    model_json = model.to_json()
    json_file = open(os.path.join(model_save_path, model_json_name), "w")
    json_file.write(model_json)
    json_file.close()
    # model.save_weights(os.path.join(model_save_path, model_h5_name))

    return model_json_name #, model_h5_name
