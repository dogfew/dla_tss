import torch
from torch import Tensor
from torch.nn import TripletMarginWithDistanceLoss


class TripletLossWrapper(TripletMarginWithDistanceLoss):
    def __init__(self, margin=0.35, *args, **kwargs):
        super().__init__(margin=margin, *args, **kwargs)

    def forward(
        self, anchor_embedding, positive_embedding, negative_embedding, **batch
    ) -> dict:
        return {'loss': super().forward(
            anchor_embedding, positive_embedding, negative_embedding
        )}
