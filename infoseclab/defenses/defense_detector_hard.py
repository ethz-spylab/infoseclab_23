import torch
import torch.nn as nn
import copy
from infoseclab.defenses import ResNet
from infoseclab import common
from sklearn.ensemble import RandomForestClassifier
from joblib import load


class RFDetector(ResNet):
    """
    A defense that also aims to detect adversarial examples using a random forest.
    """
    def __init__(self, device, detector_path=common.get_checkpoint_abs_path("data/forest_detector.joblib")):
        """
        :param device: the device to use for the defense
        :param detector_path: the path to the detector model
        """
        super().__init__(device)

        # save the classifier head of the model
        self.det = load(detector_path)

    def detect(self, x):
        """
        Detect adversarial examples.
        :param x: the batch of images of size (batch_size, 3, 224, 224) in the range [0, 255]
        :return: the detection of the defense of size (batch_size,). 0 means clean, 1 means adversarial.
        """
        return torch.from_numpy(self.det.predict(self.get_logits(x).detach().cpu().numpy()))
