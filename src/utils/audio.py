import torch
import math


class Audio:
    def __init__(
        self, n_fft=511, hop_length=None, win_length=None, ref_level=20, min_level=-100
    ):
        self.n_fft = n_fft
        self.hop_length = (
            hop_length if hop_length is not None else math.floor(n_fft / 4)
        )
        self.win_length = win_length if win_length is not None else n_fft
        self.ref_level = ref_level
        self.min_level = min_level

    def wav2spec(self, y):
        y = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        magnitudes = torch.abs(y)
        phase = torch.angle(y)
        S = 20.0 * torch.log10(torch.clamp(magnitudes, min=1e-5)) - self.ref_level
        S = torch.clamp(S / -self.min_level, -1.0, 0.0) + 1.0
        S, phase = S, phase
        return S, phase

    def spec2wav(self, spectrogram, phase):
        S = (torch.clamp(spectrogram, 0.0, 1.0) - 1.0) * -self.min_level
        S += self.ref_level
        S = 10.0 ** (S * 0.05)
        stft_matrix = torch.polar(S, phase)
        y = torch.istft(
            stft_matrix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=spectrogram.device),
            return_complex=False,
        )

        return y


class ComplexAudio:
    def __init__(
        self, n_fft=511, hop_length=None, win_length=None, ref_level=20, min_level=-100
    ):
        self.n_fft = n_fft
        self.hop_length = (
            hop_length if hop_length is not None else math.floor(n_fft / 4)
        )
        self.win_length = win_length if win_length is not None else n_fft
        self.ref_level = ref_level
        self.min_level = min_level

    def wav2spec(self, y):
        complex_spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        return complex_spec, complex_spec

    def spec2wav(self, complex_spec, phase=None):
        y = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=complex_spec.device),
        )
        return y
