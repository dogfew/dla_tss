import torch
import torch.nn as nn


class SiSDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SiSDRLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_audio, target_audio, **batch):
        target_truncated = target_audio[:, :pred_audio.shape[1]]
        pred_truncated = pred_audio[:, :target_truncated.shape[1]]
        target_centered = target_truncated - torch.mean(target_truncated, dim=-1, keepdim=True)
        pred_centered = pred_truncated - torch.mean(pred_truncated, dim=-1, keepdim=True)
        scaled_pred = torch.sum(target_centered * pred_centered, dim=-1, keepdim=True) * pred_centered / (
                torch.linalg.norm(pred_centered, ord=2, dim=-1, keepdim=True) ** 2 + self.eps)
        loss = -  20 * torch.log10(
            (torch.linalg.norm(scaled_pred, ord=2, dim=-1) + self.eps) / (
                    torch.linalg.norm(target_centered - scaled_pred, ord=2, dim=-1) + self.eps)
        )
        return {'loss': loss.mean()}


class SpExPlusLoss(nn.Module):
    def __init__(self, gamma=10):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.si_sdr = SiSDRLoss()
        self.gamma = gamma

    def mask(self, x, lengths, fill=0):
        mask = (torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) > lengths.unsqueeze(1))
        x.masked_fill_(mask, fill)

    def forward(self, s1, s2, s3, target_audio, speaker_pred,target_audio_len, **batch):
        self.mask(s1, target_audio_len)
        self.mask(s2, target_audio_len)
        self.mask(s3, target_audio_len)
        self.mask(target_audio, target_audio_len)

        snr_loss = (0.8 * self.si_sdr(s1, target_audio)['loss'] +
                    0.1 * self.si_sdr(s2, target_audio)['loss'] +
                    0.1 * self.si_sdr(s3, target_audio)['loss'])
        if self.training:
            ce_loss = self.ce_loss(speaker_pred, batch["speaker_target"])
        else:
            ce_loss = torch.tensor([0], device=s1.device)
        return {'loss': snr_loss + ce_loss * self.gamma,
                'ce_loss': ce_loss,
                'snr_loss': snr_loss
                }
