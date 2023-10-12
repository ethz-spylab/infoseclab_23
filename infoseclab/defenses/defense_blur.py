from infoseclab.defenses import ResNet
from PIL import Image, ImageFilter
import numpy as np
import torch


class ResNetBlur(ResNet):
    """
    A defense that applies a blur filter before classifying.
    """
    def __init__(self, device):
        super().__init__(device)

    def get_logits(self, x):
        return super().get_logits(blur_images(x))


def _blur(im):
    """
    Blur a torch tensor image using PIL's MedianFilter.
    :param im: the torch tensor image to convert, in the range [0, 255]
    :return: the converted image as a torch tensor, in the range [0, 255]
    """
    # convert torch tensor to PIL Image
    device = im.device
    im = Image.fromarray(np.uint8(im.cpu().numpy().transpose(1, 2, 0)))

    im = im.filter(ImageFilter.MedianFilter(size = 3))

    # convert PIL Image to torch tensor
    im = (np.asarray(im).astype(np.float32)).transpose(2, 0, 1)
    return torch.from_numpy(im).to(device)


def blur_images(images):
    """
    ConBlurvert a batch of images.
    :param images: the images to convert as a torch tensor of dimension (N, 3, 224, 224), in the range [0, 255]
    :return: the converted images as a torch tensor of dimension (N, 3, 224, 224), in the range [0, 255]
    """
    with torch.no_grad():
        return torch.stack([_blur(im) for im in images])
