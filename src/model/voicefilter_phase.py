import torch
import torch.nn as nn
import torch.nn.functional as F
from .voicefilter import ADvector, VoiceFilterBig
from math import pi, tau


class VoiceFilterPhase(VoiceFilterBig):
    def __init__(self,
                 rnn_layers: int = 2,
                 rnn_bidirectional: bool = True,
                 hidden_size: int = 256,
                 input_size: int = 256,
                 embedder_phase_path: str = None,
                 embedder_path: str = None,
                 specnet_path: str = None,
                 **kwargs):
        super().__init__(rnn_layers=rnn_layers,
                         rnn_bidirectional=rnn_bidirectional,
                         hidden_size=hidden_size,
                         input_size=input_size,
                         embedder_path=embedder_path,
                         **kwargs)
        self.lstm = nn.GRU(
            input_size * 9, input_size // 2,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional)

        self.conv = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # self.conv = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_size, kernel_size=1),
        #     nn.BatchNorm1d(hidden_size)
        # )
        # self.conv_half = nn.Sequential(
        #     nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
        #     nn.BatchNorm1d(256),
        # )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(1),
            nn.Hardtanh(-tau, tau),
        )
        self.trained_spec = False
        self.input_size = input_size
        if specnet_path is not None:
            state_dict = torch.load(specnet_path)['state_dict']
            self.specnet = VoiceFilterBig()
            self.specnet.load_state_dict(state_dict)
            self.trained_spec = True

        self.trained_embedder_phase = False
        if embedder_phase_path is not None:
            state_dict = torch.load(embedder_phase_path)['state_dict']
            self.embedder_phase = ADvector()
            self.embedder_phase.load_state_dict(state_dict)
            self.trained_embedder_phase = True
        else:
            self.embedder_phase = ADvector()

        if embedder_path is not None:
            state_dict = torch.load(embedder_path)['state_dict']
            self.specnet.embedder = ADvector()
            self.specnet.embedder.load_state_dict(state_dict)
            self.specnet.trained_embedder = True
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.zeros_(m.bias)
    #             nn.init.normal_(m.weight, mean=1.0, std=0.001)

    def forward(self, mix_phase, reference_phase, **batch):
        x = self.conv(mix_phase.unsqueeze(dim=1)).squeeze(dim=1)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        # x = self.conv(mix_phase)
        if self.trained_embedder_phase:
            with torch.no_grad():
                dvec = self.embedder.make_embedding(reference_phase.transpose(1, 2))
        else:
            dvec = self.embedder(reference_phase)
        dvec = dvec.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = torch.cat((x, dvec), dim=1)
        x, _ = self.lstm(x.transpose(1, 2))
        # VER 1
        # soft_mask_phase = self.fc(x).transpose(1, 2) * tau - pi
        # END VER 1
        # # VER 2
        # x = self.conv_half(x)
        soft_mask_phase = self.conv2(x.unsqueeze(dim=1)).squeeze(dim=1).transpose(1, 2)
        # # END VER 2
        pred_phase = soft_mask_phase + mix_phase
        pred_phase = torch.fmod(pred_phase + pi, tau) - pi
        out = {'pred_phase': pred_phase,
               'pred_mask': soft_mask_phase,
               'embed_phase': dvec}
        if self.trained_spec:
            with torch.no_grad():
                out.update(self.specnet.forward(**batch))
        return out
