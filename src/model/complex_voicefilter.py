import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.voicefilter import ADvector


class ADvectorComplex(nn.Module):
    def __init__(
            self,
            input_size=256,
            hidden_size=256,
            num_layers=1,
    ):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dtype=torch.complex64)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size, input_size, dtype=torch.complex64),
            nn.Tanh()
        )
        self.linear = nn.Linear(input_size, 1, dtype=torch.complex64)

    def make_embedding(self, inputs):
        x, _ = self.rnn(inputs)
        embeds = self.embedding(x)
        attn_weights = F.softmax(self.linear(embeds).abs(), dim=1).to(torch.complex64)
        embeds = torch.bmm(embeds.transpose(1, 2), attn_weights).squeeze(-1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

    def forward(self, anchor, positive, negative, **batch):
        anchor_embedding = self.make_embedding(anchor)
        positive_embedding = self.make_embedding(positive)
        negative_embedding = self.make_embedding(negative)
        return {'anchor_embedding': anchor_embedding,
                'positive_embedding': positive_embedding,
                'negative_embedding': negative_embedding}


class ComplexReLU(nn.Module):
    def forward(self, inp):
        return F.relu(inp.real).type(torch.complex64) + 1j * F.relu(inp.imag).type(
            torch.complex64
        )
def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
        + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, inp):
        return apply_complex(self.fc_r, self.fc_i, inp)

class VoiceFilterComplex(nn.Module):
    def __init__(self,
                 rnn_layers: int = 1,
                 rnn_bidirectional: bool = True,
                 hidden_size: int = 256,
                 input_size: int = 256,
                 embedder_path: str = None,
                 **kwargs):
        super().__init__()
        self.trained_embedder = False
        # if embedder_path is not None:
        #     state_dict = torch.load(embedder_path)['state_dict']
        #     self.embedder = ADvector()
        #     self.embedder.load_state_dict(state_dict)
        #     self.trained_embedder = True
        # else:
        self.embedder = nn.RNN(input_size, hidden_size, 2, batch_first=True, dtype=torch.complex64)
        self.lstm = nn.RNN(
            input_size * 2,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional, dtype=torch.complex64)

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1, dtype=torch.complex64),
            # nn.BatchNorm1d(hidden_size, dtype=torch.complex64)
        )
        self.fc = nn.Sequential(
            # ComplexReLU(),
            ComplexLinear(hidden_size * (2 if rnn_bidirectional else 1), hidden_size),
            # ComplexReLU(),
            ComplexLinear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, mix_spectrogram, reference_spectrogram, **batch):
        # B x C x T
        # B x 1 x C x T
        x = self.conv(mix_spectrogram)
        # # B x 8 x C x T
        # # B x 8 * C x T
        # if self.trained_embedder:
        #     with torch.no_grad():
        #         dvec = self.embedder.make_embedding(reference_spectrogram.transpose(1, 2).abs())
        # else:

        dvec, _ = self.embedder(reference_spectrogram.transpose(1, 2))
        dvec = dvec[:, -1, :]
        dvec = dvec.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = torch.cat((x, dvec), dim=1)  # B x T x C
        x, _ = self.lstm(x.transpose(1, 2))
        soft_mask = self.fc(x).transpose(1, 2)  # B x C x T
        return {'pred_spectrogram': soft_mask * mix_spectrogram,
                'pred_mask': soft_mask,
                'embed': dvec}
