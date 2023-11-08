import torch
import torch.nn as nn
import torch.nn.functional as F


class ADvector(nn.Module):
    def __init__(
        self,
        input_size=256,
        hidden_size=256,
        num_layers=3,
    ):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.embedding = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Tanh())
        self.attention = nn.Sequential(nn.Linear(input_size, 1), nn.Softmax(dim=1))

    def make_embedding(self, inputs):
        x, _ = self.rnn(inputs)
        embeds = self.embedding(x)
        attn_weights = self.attention(embeds)
        embeds = torch.bmm(embeds.transpose(1, 2), attn_weights).squeeze(-1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

    def forward(self, anchor, positive, negative, **batch):
        anchor_embedding = self.make_embedding(anchor)
        positive_embedding = self.make_embedding(positive)
        negative_embedding = self.make_embedding(negative)
        return {
            "anchor_embedding": anchor_embedding,
            "positive_embedding": positive_embedding,
            "negative_embedding": negative_embedding,
        }


class VoiceFilter(nn.Module):
    def __init__(
        self,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        hidden_size: int = 256,
        input_size: int = 256,
        embedder_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.trained_embedder = False
        if embedder_path is not None:
            state_dict = torch.load(embedder_path)["state_dict"]
            self.embedder = ADvector()
            self.embedder.load_state_dict(state_dict)
            self.trained_embedder = True
        else:
            self.embedder = ADvector()
        self.lstm = nn.GRU(
            input_size * 2,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
        )
        self.dropout1 = nn.Dropout2d(p=0.25)
        # self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * (2 if rnn_bidirectional else 1), hidden_size),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, mix_spectrogram, reference_spectrogram, **batch):
        # B x C x T
        # B x 1 x C x T
        x = self.conv(mix_spectrogram)
        # # B x 8 x C x T
        # # B x 8 * C x T
        if self.trained_embedder:
            with torch.no_grad():
                dvec = self.embedder.make_embedding(
                    reference_spectrogram.transpose(1, 2)
                )
        else:
            dvec = self.embedder(reference_spectrogram)
        dvec = dvec.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = torch.cat((x, dvec), dim=1)  # B x T x C
        x = self.dropout1(x)
        x, _ = self.lstm(x.transpose(1, 2))
        soft_mask = self.fc(x).transpose(1, 2)  # B x C x T
        return {
            "pred_spectrogram": soft_mask * mix_spectrogram,
            "pred_mask": soft_mask,
            "embed": dvec,
        }


class VoiceFilterBig(nn.Module):
    def __init__(
        self,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        hidden_size: int = 400,
        input_size: int = 256,
        embedder_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.trained_embedder = False
        if embedder_path is not None:
            state_dict = torch.load(embedder_path)["state_dict"]
            self.embedder = ADvector()
            self.embedder.load_state_dict(state_dict)
            self.trained_embedder = True
        else:
            self.embedder = ADvector()
        self.lstm = nn.GRU(
            input_size * 9,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )

        self.conv = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
        )
        self.dropout = nn.Dropout2d(0.25)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * (2 if rnn_bidirectional else 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, mix_spectrogram, reference_spectrogram, **batch):
        # B x C x T
        # B x 1 x C x T
        x = mix_spectrogram.unsqueeze(dim=1)
        x = self.conv(x).squeeze(dim=1)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        # # B x 8 x C x T
        # # B x 8 * C x T
        if self.trained_embedder:
            with torch.no_grad():
                dvec = self.embedder.make_embedding(
                    reference_spectrogram.transpose(1, 2)
                )
        else:
            dvec = self.embedder(reference_spectrogram)
        dvec = dvec.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = torch.cat((x, dvec), dim=1)  # B x T x C
        x, _ = self.lstm(x.transpose(1, 2))
        x = self.dropout(x)
        soft_mask = self.fc(x).transpose(1, 2)  # B x C x T
        return {
            "pred_spectrogram": soft_mask * mix_spectrogram,
            "pred_mask": soft_mask,
            "embed": dvec,
        }
