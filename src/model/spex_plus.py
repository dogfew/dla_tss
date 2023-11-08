import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(GlobalLayerNorm, self).__init__()
        self.num_features = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1))

    def forward(self, x):
        mean = x.mean(dim=[1, 2], keepdim=True)
        std = x.std(dim=[1, 2], keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = self.weight * x + self.bias
        return x


class TCNBlock(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, conv_channels, 1)
        self.net = nn.Sequential(
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(
                conv_channels,
                conv_channels,
                kernel_size,
                groups=conv_channels,
                padding=(dilation * (kernel_size - 1)) // 2,
                dilation=dilation,
            ),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels),
            nn.Conv1d(conv_channels, in_channels, 1),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.net(y)
        return y + x


class TCNBlockSpeaker(TCNBlock):
    def __init__(
        self,
        in_channels=256,
        embedd_dim=100,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
    ):
        super().__init__(in_channels, conv_channels, kernel_size, dilation)
        self.conv = nn.Conv1d(in_channels + embedd_dim, conv_channels, 1)

    def forward(self, x, aux):
        aux = aux.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        y = torch.cat([x, aux], 1)
        y = self.conv(y)
        y = self.net(y)
        return y + x


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.PReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
        )

        self.downsample = (
            nn.Sequential(nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False))
            if input_dim != output_dim
            else nn.Identity()
        )

        self.second = nn.Sequential(nn.PReLU(), nn.MaxPool1d(3))

    def forward(self, x):
        y = self.first(x)
        x = self.downsample(x)
        y += x
        y = self.second(y)
        return y


class SharedEncoder(nn.Module):
    def __init__(self, N, L1, L2, L3):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_1d_short = nn.Sequential(
            nn.Conv1d(1, N, L1, stride=L1 // 2), nn.ReLU()
        )
        self.encoder_1d_middle = nn.Sequential(
            nn.Conv1d(1, N, L2, stride=L1 // 2), nn.ReLU()
        )
        self.encoder_1d_long = nn.Sequential(
            nn.Conv1d(1, N, L3, stride=L1 // 2), nn.ReLU()
        )

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(dim=1)
        z1 = self.encoder_1d_short(x_unsqueezed)
        padding_const = (z1.shape[-1] - 1) * (self.L1 // 2) - x.shape[-1]
        padding_mid = padding_const + self.L2
        padding_long = padding_const + self.L3
        z2 = self.encoder_1d_middle(F.pad(x_unsqueezed, (0, padding_mid)))
        z3 = self.encoder_1d_long(F.pad(x_unsqueezed, (0, padding_long)))
        return z1, z2, z3


class StackedTCNs(
    nn.Module,
):
    def __init__(
        self, embedd_dim, in_channels, conv_channels, kernel_size, num_blocks=4
    ):
        super().__init__()
        self.speaker_tcn = TCNBlockSpeaker(
            embedd_dim=embedd_dim,
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dilation=1,
        )

        self.stacked_tcns = nn.Sequential(
            *[
                TCNBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    dilation=(2**b),
                )
                for b in range(1, num_blocks)
            ]
        )

    def forward(self, x, aux):
        y = self.speaker_tcn(x, aux)
        y = self.stacked_tcns(y)
        return y


class SpExPlus(nn.Module):
    def __init__(
        self,
        L1=0.0025,
        L2=0.0100,
        L3=0.0200,
        N=256,
        B=8,
        O=256,
        P=512,
        Q=3,
        R=4,
        num_speakers=56,
        embed_dim=256,
        sample_rate=16_000,
    ):
        super().__init__()
        L1, L2, L3 = int(L1 * sample_rate), int(L2 * sample_rate), int(L3 * sample_rate)
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.shared_encoder = SharedEncoder(N, L1, L2, L3)
        self.layer_norm_y = ChannelwiseLayerNorm(3 * N)
        self.cnn1x1_y = nn.Conv1d(3 * N, O, 1)
        self.tcns = nn.ModuleList(
            [
                StackedTCNs(
                    embedd_dim=embed_dim,
                    in_channels=O,
                    conv_channels=P,
                    kernel_size=Q,
                    num_blocks=B,
                )
                for _ in range(R)
            ]
        )
        self.mask1 = nn.Sequential(nn.Conv1d(O, N, 1), nn.ReLU())
        self.mask2 = nn.Sequential(nn.Conv1d(O, N, 1), nn.ReLU())
        self.mask3 = nn.Sequential(nn.Conv1d(O, N, 1), nn.ReLU())
        ### Speech Decoder
        self.decoder_short = nn.ConvTranspose1d(N, 1, kernel_size=L1, stride=L1 // 2)
        self.decoder_middle = nn.ConvTranspose1d(N, 1, kernel_size=L2, stride=L1 // 2)
        self.decoder_long = nn.ConvTranspose1d(N, 1, kernel_size=L3, stride=L1 // 2)
        self.num_speakers = num_speakers
        ### Speaker Encoder
        self.layer_norm_x = ChannelwiseLayerNorm(3 * N)
        self.speaker_encoder = nn.Sequential(
            nn.Conv1d(3 * N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            nn.Conv1d(P, embed_dim, 1),
        )
        self.speaker_linear = nn.Linear(embed_dim, num_speakers)

    def forward(self, mix_audio, reference_audio, reference_audio_len, **batch):
        cutting = mix_audio.shape[-1]
        y1, y2, y3 = self.shared_encoder(mix_audio)
        y = self.layer_norm_y(torch.cat([y1, y2, y3], 1))
        y = self.cnn1x1_y(y)

        x1, x2, x3 = self.shared_encoder(reference_audio)
        x = self.layer_norm_x(torch.cat([x1, x2, x3], 1))
        x = self.speaker_encoder(x)
        # Mean Pooling
        v = x.sum(-1) / (
            ((reference_audio_len - self.L1) // (self.L1 // 2) + 1) // 27
        ).unsqueeze(1)

        for tcn in self.tcns:
            y = tcn(y, v)

        m1 = self.mask1(y)
        m2 = self.mask2(y)
        m3 = self.mask3(y)
        s1 = self.decoder_short(y1 * m1).squeeze(dim=1)
        s2 = self.decoder_middle(y2 * m2)[:, :cutting].squeeze(dim=1)
        s3 = self.decoder_long(y3 * m3)[:, :cutting].squeeze(dim=1)
        return {"s1": s1, "s2": s2, "s3": s3, "speaker_pred": self.speaker_linear(v)}
