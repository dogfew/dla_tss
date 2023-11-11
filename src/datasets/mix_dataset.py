import logging
import torchaudio
import random
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from pathlib import Path
from src.utils.parse_config import ConfigParser


class MixDataset:
    def __init__(
        self,
        dir,
        config_parser: ConfigParser,
        audio,
        cut_reference=None,
        *args,
        **kwargs,
    ):
        self.config_parser = config_parser
        self.speaker_lst = []
        self.cut_reference = cut_reference
        self.speaker_audio_dict = {}
        self.audio = audio
        data = []
        for audio_file in Path(dir).rglob("*-target.[mwflac4]*"):
            base_key = audio_file.stem[: -len("-target")]
            suffix = audio_file.suffix
            mixed_file = audio_file.parent / f"{base_key}-mixed{suffix}"
            ref_file = audio_file.parent / f"{base_key}-ref{suffix}"
            if mixed_file.exists() and ref_file.exists():
                speaker_target, speaker_noise, *_ = base_key.split("_")
                data.append(
                    {
                        "speaker_target": speaker_target,
                        "speaker_noise": speaker_noise,
                        "target_path": str(audio_file),
                        "mix_path": str(mixed_file),
                        "reference_path": str(ref_file),
                    }
                )
                if speaker_target not in self.speaker_audio_dict:
                    self.speaker_lst.append(speaker_target)
                    self.speaker_audio_dict[speaker_target] = []
                self.speaker_audio_dict[speaker_target].append(str(ref_file))
        self.data = data
        self._index = self.data

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        target_audio_wave, _ = torchaudio.load(data_dict["target_path"])
        mix_audio_wave, _ = torchaudio.load(data_dict["mix_path"])
        reference_audio_wave, _ = torchaudio.load(data_dict["reference_path"])
        if self.cut_reference:
            target_length = mix_audio_wave.size(1)
            current_length = reference_audio_wave.size(1)
            if current_length > target_length:
                reference_audio_wave = reference_audio_wave[:, :target_length]
            elif current_length < target_length:
                padding = target_length - current_length
                reference_audio_wave = F.pad(
                    reference_audio_wave, (0, padding), "constant", 0
                )

        speaker_target = self.speaker_lst.index(data_dict["speaker_target"])
        return {
            "speaker_target": speaker_target,
            "target_audio": target_audio_wave,
            "mix_audio": mix_audio_wave,
            "reference_audio": reference_audio_wave,
            "target_audio_path": data_dict["target_path"],
        }

    def __len__(self):
        return len(self._index)


class MixDatasetVoiceFilter(MixDataset):
    def __getitem__(self, ind):
        data_dict = self._index[ind]
        target_audio_wave, _ = torchaudio.load(data_dict["target_path"])
        mix_audio_wave, _ = torchaudio.load(data_dict["mix_path"])
        reference_audio_wave, _ = torchaudio.load(data_dict["reference_path"])
        speaker_target = self.speaker_lst.index(data_dict["speaker_target"])
        target_audio_spec, target_phase = self.audio.wav2spec(target_audio_wave)
        mix_audio_spec, mix_phase = self.audio.wav2spec(mix_audio_wave)
        reference_audio_spec, reference_phase = self.audio.wav2spec(
            reference_audio_wave
        )
        return {
            "speaker_target": speaker_target,
            "target_audio": target_audio_wave,
            "reference_audio": reference_audio_wave,
            "mix_audio": mix_audio_wave,
            "mix_phase": mix_phase,
            "target_phase": target_phase,
            "reference_phase": reference_phase,
            "target_spectrogram": target_audio_spec,
            "mix_spectrogram": mix_audio_spec,
            "reference_spectrogram": reference_audio_spec,
            "target_audio_path": data_dict["target_path"],
        }

    def __len__(self):
        return len(self._index)


class TripletAudioDataset(MixDataset):
    def __init__(self, use_phase=False, *args, **kwargs):
        self.idx = 0 if use_phase else 1
        super().__init__(*args, **kwargs)

    def __getitem__(self, ind):
        anchor_speaker = self.speaker_lst[ind]
        anchor_audios = self.speaker_audio_dict[anchor_speaker]

        if len(anchor_audios) >= 2:
            anchor_audio_path, positive_audio_path = random.sample(anchor_audios, 2)
        else:
            anchor_audio_path, positive_audio_path = anchor_audios[0], anchor_audios[0]
        anchor_audio_wave, _ = torchaudio.load(anchor_audio_path)
        positive_audio_wave, _ = torchaudio.load(positive_audio_path)

        anchor_audio_spec = self.audio.wav2spec(anchor_audio_wave)[self.idx]
        positive_audio_spec = self.audio.wav2spec(positive_audio_wave)[self.idx]
        negative_speaker = random.choice(
            [s for s in self.speaker_lst if s != anchor_speaker]
        )
        negative_audios = self.speaker_audio_dict[negative_speaker]

        negative_audio_path = random.choice(negative_audios)
        negative_audio_wave, _ = torchaudio.load(negative_audio_path)
        negative_audio_spec = self.audio.wav2spec(negative_audio_wave)[self.idx]
        return {
            "anchor": anchor_audio_spec,
            "positive": positive_audio_spec,
            "negative": negative_audio_spec,
            "speaker_target": int(anchor_speaker),
        }

    def __len__(self):
        return len(self.speaker_lst)
