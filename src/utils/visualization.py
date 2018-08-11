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


def show_train_stats(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    plt.show()