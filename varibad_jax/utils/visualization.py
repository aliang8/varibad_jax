import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def custom_to_pil(x):
    x = np.clip(x, 0.0, 1.0)
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def plot_images(images):
    fig = plt.figure(figsize=(40, 20))
    columns = 4
    rows = 2
    plt.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()


def stack_reconstructions(images):
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images) * w, h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i * w, 0))
    return img


def make_image_grid(images, num_rows=2):
    """
    images is a list of PIL images
    """
    num_cols = len(images) // num_rows
    assert num_cols * num_rows == len(images)

    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (num_cols * w, num_rows * h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i % num_cols * w, i // num_cols * h))
    return img
