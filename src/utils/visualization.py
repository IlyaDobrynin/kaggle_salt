import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

def show_images(in_df, img_size):
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
        ax.text(1, img_size - 1, in_df.loc[idx].z, color="black")
        ax.text(img_size - 1, 1, round(in_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
        ax.text(1, 1, in_df.loc[idx].coverage_class, color="black", ha="left", va="top")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")
    plt.show()


def show_images_from_npy(test_npy_path, preds_npy_path, num_images):
    images_array = np.load(test_npy_path)
    masks_array = np.load(preds_npy_path)
    #
    # print(images_array.shape)
    # print(masks_array.shape)

    grid_width = 10
    grid_height = int(num_images/(grid_width/2))

    # fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    images_numbers_array = np.random.randint(0, images_array.shape[0], num_images)
    for i, idx in enumerate(images_numbers_array):
    #     print(i, ":", idx)
    #     img = np.squeeze(images_array[idx, :, :, :], axis=-1)
        mask = np.squeeze(masks_array[idx, :, :, :], axis=-1)
        img = images_array[idx, :, :]
        # mask = masks_array[idx, :, :]
        fig = plt.figure(1, figsize=(15, 15))
        print(img.shape, mask.shape)
        ax = fig.add_subplot(grid_height, grid_width, (i+1) * 2 - 1)
        ax.imshow(img)
        ay = fig.add_subplot(grid_height, grid_width, (i+1) * 2)
        ay.imshow(mask)

    plt.show()


def show_train_stats(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    plt.show()


if __name__ == '__main__':
    test_path = r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\in_images\test.npy'
    preds_path = r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\out_images\ding_han\ding_han_test.npy'

    train_images_path = r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\in_images\train_pad.npy'
    train_masks_path = r'E:\Programming\ML Competitions\kaggle_salt_challenge_data\in_images\train_masks.npy'

    show_images_from_npy(test_npy_path=train_images_path,
                         preds_npy_path=train_masks_path,
                         num_images=20)
