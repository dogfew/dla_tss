import torch
from torch import Tensor
from torch.nn import L1Loss


class L1LossWrapper(L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred_spectrogram, target_spectrogram, **batch):
        return {"loss": super().forward(pred_spectrogram, target_spectrogram)}
