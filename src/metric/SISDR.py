from src.base.base_metric import BaseMetric
from src.metric.utils import si_sdr
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SiSDR(ScaleInvariantSignalDistortionRatio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SiSDR"

    def __call__(self, pred_audio, target_audio, **batch):
        sisdr = super().__call__(pred_audio, target_audio)
        return sisdr


class SiSDREsts(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        si_sdrs = []
        for pred_audio0, pred_audio1, pred_audio2, target_audio in zip(
            batch["s1"], batch["s2"], batch["s3"], batch["target_audio"]
        ):
            pred_audio0 = self.process_audio(pred_audio0, target_audio.shape[0])
            pred_audio = pred_audio0
            si_sdrs.append(si_sdr(pred_audio, target_audio))
        return sum(si_sdrs) / len(si_sdrs)

    def process_audio(self, est, target_len):
        if est.size(0) > target_len:
            est = est[:target_len]
        elif est.size(0) < target_len:
            pad_size = target_len - est.size(0)
            est = F.pad(est, (0, pad_size), "constant", 0)
        return est
