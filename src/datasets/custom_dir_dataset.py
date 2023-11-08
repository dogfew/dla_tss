import torchaudio

from pathlib import Path
import torch.nn.functional as F


class CustomDirDataset:
    def __init__(self, dir, *args, **kwargs):
        data = []
        path_dir = Path(dir)
        for audio_file in (path_dir / "targets").rglob("*-target.[mwflac4]*"):
            base_key = audio_file.stem[: -len("-target")]
            suffix = audio_file.suffix
            mixed_file = path_dir / "mix" / f"{base_key}-mixed{suffix}"
            ref_file = path_dir / "refs" / f"{base_key}-ref{suffix}"
            if mixed_file.exists() and ref_file.exists():
                data.append(
                    {
                        "target_path": str(audio_file),
                        "mix_path": str(mixed_file),
                        "reference_path": str(ref_file),
                    }
                )
        self.data = data

    def __getitem__(self, ind):
        data_dict = self.data[ind]
        target_audio_wave, _ = torchaudio.load(data_dict["target_path"])
        mix_audio_wave, _ = torchaudio.load(data_dict["mix_path"])
        reference_audio_wave, _ = torchaudio.load(data_dict["reference_path"])
        return {
            "target_audio": target_audio_wave,
            "mix_audio": mix_audio_wave,
            "reference_audio": reference_audio_wave,
            "target_audio_path": data_dict["target_path"],
        }

    def __len__(self):
        return len(self.data)
