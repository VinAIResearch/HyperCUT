import importlib
import os.path as osp
import random

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def is_scalar(var):
    return not isinstance(var, (tuple, list, dict, set, np.ndarray))


def load_module_from_file(path):
    module_name = osp.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def visualize_tsne(feats, labels, save_path):
    feats_embeded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(feats)

    df = pd.DataFrame({"x": feats_embeded[:, 0], "y": feats_embeded[:, 1], "f": labels})

    color_palette = sns.color_palette("Spectral", as_cmap=True)
    plot = sns.scatterplot(data=df, x="x", y="y", hue="f", palette=color_palette)
    plot.legend(title="$\mathcal{H}$", fontsize=12, title_fontsize=15)
    plot.tick_params(axis="both", which="major", labelsize=12)
    fig = plot.get_figure()

    fig.savefig(save_path, dpi=300)
    fig.clf()


class NearBBoxResizedSafeCrop:
    """
    Crop near the outside of the bbox and resize the cropped image
    max_ratio means the max valid crop ratio from boundary of the bbox to boundary of the image, (0, 1)
    bbox format is pascal_voc, a bounding box looks like [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212]
    """

    def __init__(self, height, width, max_ratio=0.1):
        self.height = height
        self.width = width
        self.max_ratio = max_ratio

    def __call__(self, image, prev_img, bbox, flow=False, trend=False, args=None):
        if len(bbox) == 5:
            x_min, y_min, x_max, y_max, _ = bbox
        else:
            x_min, y_min, x_max, y_max = bbox

        img_h, img_w, _ = prev_img.shape

        # Prepare args
        if args is None:
            args = {}
            args["ratio"] = self.max_ratio * random.uniform(0, 1)

        # Crop image
        ratio = args["ratio"]
        x_min = int((1 - ratio) * x_min)
        y_min = int((1 - ratio) * y_min)
        x_max = int(x_max + ratio * (img_w - x_max))
        y_max = int(y_max + ratio * (img_h - y_max))
        image = image[y_min:y_max, x_min:x_max]
        if trend:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
        else:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if flow:
            image[:, :, 0] *= self.width / float(x_max - x_min)
            image[:, :, 1] *= self.height / float(y_max - y_min)

        # args: arguments for replaying this augmentation
        # todo: re-calculate the bbox
        return image
