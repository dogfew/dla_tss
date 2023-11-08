import torch
from torch import Tensor
from torch.nn import MSELoss


class MSELossWrapper(MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_spectrogram, target_spectrogram, **batch) -> Tensor:
        # loss = super().forward(
        #     pred_spectrogram,
        #     target_spectrogram
        # )
        loss = super().forward(
            pred_spectrogram, target_spectrogram
        ) + 0.2 * super().forward((pred_spectrogram).abs(), (target_spectrogram).abs())
        # decay = - 0.05 * super().forward(batch['mix_spectrogram'], pred_spectrogram)
        return {"loss": loss}
