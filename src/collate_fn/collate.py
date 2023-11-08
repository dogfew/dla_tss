import logging
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict]) -> dict:
    """
    Collate and pad fields in dataset items
    """
    result_batch = dict()
    spectrogram_keys = {
        "target_spectrogram",
        "mix_spectrogram",
        "reference_spectrogram",
    }
    audio_keys = {"target_audio", "mix_audio", "reference_audio"}
    phase_keys = {"mix_phase", "target_phase", "reference_phase"}
    for key in dataset_items[0].keys():
        items = [item[key] for item in dataset_items]
        if key in audio_keys or key in spectrogram_keys:
            result_batch[f"{key}_len"] = torch.tensor(
                data=[item.shape[-1] for item in items]
            )
            result_batch[key] = pad_sequence(
                sequences=[item.T for item in items],
                batch_first=True,
            ).squeeze(dim=2)
        elif key in phase_keys:
            result_batch[key] = pad_sequence(
                sequences=[torch.squeeze(item, dim=0).t() for item in items],
                batch_first=True,
            )
        else:
            result_batch[key] = [item.get(key) for item in dataset_items]
    if "mix_phase" in result_batch:
        result_batch["mix_phase"] = result_batch["mix_phase"].transpose(2, 1)
        result_batch["reference_phase"] = result_batch["reference_phase"].transpose(
            2, 1
        )
        result_batch["target_phase"] = result_batch["target_phase"].transpose(2, 1)
        result_batch["target_spectrogram"] = result_batch["target_spectrogram"].permute(
            0, 2, 1
        )
        result_batch["mix_spectrogram"] = result_batch["mix_spectrogram"].permute(
            0, 2, 1
        )
        result_batch["reference_spectrogram"] = result_batch[
            "reference_spectrogram"
        ].permute(0, 2, 1)
    result_batch["speaker_target"] = torch.tensor(result_batch["speaker_target"])
    return result_batch


def dvec_collate_fn(dataset_items: list[dict]) -> dict:
    result_batch = dict()
    for key in dataset_items[0].keys():
        if key != "speaker_target":
            result_batch[key] = pad_sequence(
                sequences=[
                    torch.squeeze(item.get(key), dim=0).t() for item in dataset_items
                ],
                batch_first=True,
            )
        else:
            result_batch[key] = [item[key] for item in dataset_items]
    result_batch["speaker_target"] = torch.tensor(result_batch["speaker_target"])
    return result_batch
