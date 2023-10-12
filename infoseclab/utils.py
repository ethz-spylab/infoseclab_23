import warnings
import inspect
import os
import torch
import numpy as np
from infoseclab.data import ImageNet, th_to_npy_uint8, npy_uint8_to_th
import matplotlib.pyplot as plt
from tqdm import trange


def batched_func(func, inputs, device, batch_size=20, disable_tqdm=False, **kwargs):
    """
    Apply a function to a batch of inputs.
    :param func: the function to apply
    :param inputs: a tuple of inputs to the function
    :param device: the device that inputs should be moved to
    :param batch_size: the batch size
    :param disable_tqdm: whether to disable the progress bar
    :param kwargs: additional keyword arguments to pass to the function
    :return: the outputs of the function over all inputs
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    n = len(inputs[0])

    num_batches = int((n + batch_size - 1) / batch_size)

    outputs = []
    for i in trange(num_batches, disable=disable_tqdm):
        batch = [x[i * batch_size:(i + 1) * batch_size].to(device) for x in inputs]
        outputs.append(func(*batch, **kwargs).detach().cpu())

    return torch.cat(outputs, dim=0)


def save_images(path, images):
    """
    Save images as uint8 to a numpy file.
    :param path: the path to the file
    :param images: the images to save as a torch tensor of dimension (N, 3, 224, 224)
    """
    images_npy = th_to_npy_uint8(images)

    images_th = npy_uint8_to_th(images_npy)
    if not torch.allclose(images, images_th):
        warnings.warn("Images are not the same after saving and loading.")

    np.save(path, images_npy)


def save_attack_scores(path, scores):
    """
    Save attack scores as float32 to a numpy file.
    :param path: the path to the file
    :param scores: the attack scores to save as a numpy array of dimension (N,)
    """
    if scores.shape != (50000,):
      raise ValueError(
        f"scores must be an array of shape (50000,) but was {scores.shape}"
      )
    if scores.dtype != np.float32:
      raise ValueError(
        f"scores must be a float32 array, but was {scores.dtype}"
      )

    np.save(path, scores)


def img2plt(img):
    """
    Convert an image from a torch tensor to a numpy array for plotting.
    :param img: the torch image
    :return: a numpy image for plotting
    """
    assert len(img.shape) in [2, 3]
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)

    if (len(img.shape) == 3) and (img.shape[0] in [3, 1]):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img[:, :]

    return img.astype(np.uint8)


def display(image, image_orig=None, logits=None, class_names=ImageNet.class_names):
    """
    Display an adversarial example and its predicted label.
    :param image: the adversarial example
    :param image_orig: the clean image
    :param logits: the logits of the input images of dimension (1, 1000) or (2, 1000)
    :param class_names: the label names
    """
    image = img2plt(image)
    if image_orig is not None:
        image_orig = img2plt(image_orig)

    if logits is not None:
        assert len(logits.shape) == 2
        if image_orig is None:
            assert logits.shape[0] == 1
        else:
            assert logits.shape[0] == 2

    cmap = None
    if len(image.shape) == 2:
        cmap = "gray"

    if image_orig is None:
        fig, ax = plt.subplots(1)
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, 3)
        ax[1].imshow(image_orig, cmap=cmap)
        ax[1].axis('off')
        diff = image - image_orig
        ax[2].imshow((diff - np.min(diff)) / (np.max(diff) - np.min(diff) + 1e-8))
        ax[2].axis('off')
        ax[2].set_title("difference")

    ax[0].imshow(image, cmap=cmap)
    ax[0].axis('off')
    label = "adv: "
    if logits is not None:
        label += class_names[torch.argmax(logits[0])]
        confidence = torch.nn.Softmax(dim=-1)(logits[0])[torch.argmax(logits[0])]
        ax[0].set_title(f"{label}\n({confidence:.1%})")

    label = "clean: "
    if logits is not None and image_orig is not None:
        label += class_names[torch.argmax(logits[1])]
        confidence = torch.nn.Softmax(dim=-1)(logits[1])[torch.argmax(logits[1])]
        ax[1].set_title(f"{label}\n({confidence:.1%})")
    fig.tight_layout(pad=5.0)
    plt.show()

