import torch
import numpy as np
from infoseclab.defenses.defense import Defense
from infoseclab import common


class ResNetDiscrete(Defense):
    """
    A defense that uses a ResNet18 model on discretized features
    """
    def __init__(self, device, model_path=common.get_checkpoint_abs_path("data/resnet_discrete.pth")):
        super().__init__(device)
        self.model = torch.load(model_path).float()
        self.model.eval()
        self.model.to(device)

    def encode(self, x):
        assert len(x.shape) == 4
        assert x.shape[1:] == (3, 224, 224)
        assert x.min() >= 0 and x.max() <= 1.0

        # put the channels last
        x = x.permute(0, 2, 3, 1)
        shape = list(x.shape)
        
        # add a dummy dimension
        x = x.unsqueeze(-1)

        # discretize into 20 bins
        thresholds = torch.from_numpy(np.arange(0, 1, .05)).to(x.device).float()
        gt_threshold = x > thresholds
        x = gt_threshold.float()

        # reshape to original shape with 20x channels
        shape[-1] *= thresholds.shape[0]
        x = x.reshape(shape)
        x = x.permute(0, 3, 1, 2)

        assert x.shape[1:] == (60, 224, 224)
        return x

    def get_logits(self, x):
        return self.model(self.encode(x / 255.0))
