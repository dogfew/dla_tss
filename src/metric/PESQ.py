from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import pesq


class PESQ(PerceptualEvaluationSpeechQuality):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "PESQ"

    def __call__(self, pred_audio, target_audio, **batch):
        try:
            score = super().__call__(pred_audio, target_audio)
            return score
        except pesq.cypesq.NoUtterancesError:
            return -1
