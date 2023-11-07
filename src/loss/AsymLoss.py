import torch
import torch.nn as nn


class AsymmetricL2Loss(nn.Module):
    def __init__(self, alpha=10):
        super(AsymmetricL2Loss, self).__init__()
        self.alpha = alpha

    def forward(self, pred_spectrogram, target_spectrogram, **batch):
        delta_sq = (target_spectrogram - pred_spectrogram) ** 2
        loss = torch.where(target_spectrogram > pred_spectrogram,
                           delta_sq, self.alpha * delta_sq)
        return {'loss': torch.mean(loss)}


class AsymmetricL2LossPhase(AsymmetricL2Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred_phase, target_phase, **batch):
        return super().forward(pred_phase, target_phase)
