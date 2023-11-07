import editdistance
import torch
import torch.functional as F


def snr(est, target):
    return 20 * torch.log10(torch.linalg.norm(target) / (torch.linalg.norm(target - est) + 1e-6) + 1e-6)


def si_sdr(est, target):
    alpha = (target * est).sum() / torch.linalg.norm(target) ** 2
    return 20 * torch.log10(torch.linalg.norm(alpha * target) / (torch.linalg.norm(alpha * target - est) + 1e-6) + 1e-6)


def mask_preds(x, lengths, fill=0):
    max_length = lengths.max().item()
    if x.size(1) < max_length:
        pad_size = max_length - x.size(1)
        x = F.pad(x, (0, pad_size), value=fill)
    mask = torch.arange(max_length, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
    x.masked_fill_(mask, fill)
    return x
