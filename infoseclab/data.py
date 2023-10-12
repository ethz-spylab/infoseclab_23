import numpy as np
import torch
import json
from infoseclab import common

EPSILON = 8.  # Our perturbation budget is 8 (out of 256) for each pixel.


def th_to_npy_uint8(x):
    """
    Convert a torch tensor to a numpy array of uint8 values.
    :param x: a torch tensor of floats in the range [0, 255]
    :return: a numpy array of uint8 values
    """
    return np.rint(x.detach().cpu().numpy()).astype(np.uint8)


def npy_uint8_to_th(x):
    """
    Convert a numpy array of uint8 values to a torch tensor.
    :param x: a numpy array of uint8 values
    :return: a torch tensor of floats in the range [0, 255]
    """
    assert x.dtype == np.uint8
    return torch.from_numpy(x.astype(np.float32))


class ImageNet:
    """
    A subset of 200 images from the ImageNet validation set.
    """

    # get the labels corresponding to the 1000 ImageNet classes
    #with open("infoseclab/data/imagenet-simple-labels.json") as f:
    with open(common.get_checkpoint_abs_path("data/imagenet-simple-labels.json")) as f:
        class_names = json.load(f)

    # load the images and labels
    clean_images = npy_uint8_to_th(np.load(common.get_checkpoint_abs_path("data/images.npy")))
    labels = torch.from_numpy(np.load(common.get_checkpoint_abs_path("data/labels.npy")))

    # attack targets
    targets = torch.from_numpy(np.load(common.get_checkpoint_abs_path("data/targets.npy")))


class ShadowModels:
    """
    A collection of 128 shadow model predictions,
    127 of which can be used to attack the 128th model.
    """
    # Data is split into chunks of < 100MiB that need to be loaded separately
    activations_fit = []
    activations_attack = []
    labels = []
    training_splits = []
    for idx in range(5):
        data_chunk = np.load(
          common.get_checkpoint_abs_path(f"data/mia_unknown_splits_{idx}.npz")
        )
        activations_fit.append(data_chunk["activations_fit"])
        activations_attack.append(data_chunk["activations_attack"])
        labels.append(data_chunk["labels"])
        training_splits.append(data_chunk["training_splits"])
    del idx, data_chunk

    # Combine individual chunks into large arrays
    activations_fit = np.concatenate(activations_fit, axis=0)
    activations_attack = np.concatenate(activations_attack, axis=0)
    labels = np.concatenate(labels, axis=0)
    training_splits = np.concatenate(training_splits, axis=0)
