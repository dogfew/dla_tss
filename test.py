import argparse
import json
import math
import os
import pandas as pd
from pathlib import Path

import torch
import torchaudio
from torch import autocast
from tqdm import tqdm
import torch.nn.functional as F
import src.model as module_model
from src.metric.PESQ import PESQ
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from src.metric.SISDR import SiSDR, SiSDREsts

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def mask(x, lengths, fill=0):
    mask = torch.arange(x.size(1), device=x.device).expand(
        x.size(0), -1
    ) > lengths.unsqueeze(1)
    x.masked_fill_(mask, fill)
    return x


def separate_batch(batch, frame_size=16_000):
    new_batch = dict()
    mix_audio = batch['mix_audio']
    padded_audio = torch.nn.functional.pad(mix_audio, (0, frame_size - mix_audio.size(1) % frame_size))
    new_batch['mix_audio'] = padded_audio.reshape(-1, frame_size)
    target_audio = batch['target_audio']
    padded_audio = torch.nn.functional.pad(target_audio, (0, frame_size - target_audio.size(1) % frame_size))
    new_batch['target_audio'] = padded_audio.reshape(-1, frame_size)
    reference_audio = batch['reference_audio'].repeat(len(new_batch['mix_audio']), 1)
    new_batch['reference_audio'] = reference_audio
    new_batch['reference_audio_len'] = torch.tensor([reference_audio.shape[1]]).repeat(len(new_batch['mix_audio']),
                                                                                       1).flatten()
    new_batch['target_audio_len'] = batch['target_audio_len'].flatten()
    return new_batch


def main(config, args):
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(config, None)
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    metric = SiSDR(zero_mean=True).to(device)
    sample_rate = config['preprocessing']['sr']
    metricPESQ = PESQ(sample_rate, 'wb').to(device)
    with torch.no_grad():
        for test_type in ["test", "test-clean", "test-other", "val"]:
            pred_audios = []
            target_audios = []
            if test_type not in dataloaders.keys():
                continue
            for batch_num, batch in enumerate(tqdm(dataloaders[test_type])):
                if args.window_size == 0:
                    batch = Trainer.move_batch_to_device(batch, device)
                    output = model(**batch)
                    batch.update(output)
                    batch['s1'] = mask(batch['s1'], batch['mix_audio_len'])
                else:
                    batch = Trainer.move_batch_to_device(separate_batch(batch,
                                                                        frame_size=int(args.window_size * sample_rate)),
                                                         device)
                    reference_minibatches = torch.split(batch['reference_audio'], 16)
                    mix_minibatches = torch.split(batch['mix_audio'], 16)
                    reference_len_minibatches = torch.split(batch['reference_audio_len'], 16)
                    batch_results = []
                    for ref, mix, ref_len in zip(reference_minibatches, mix_minibatches, reference_len_minibatches):
                        batch_results.append(model(mix, ref, ref_len)['s1'])
                    batch['s1'] = torch.concatenate(batch_results)
                    batch['s1'] = batch['s1'].flatten().reshape(1, -1)
                    batch['target_audio'] = batch['target_audio'].flatten()[: batch['target_audio_len']].reshape(1, -1)
                if 's1' in batch:
                    # padding and cutting
                    T = batch['target_audio'].shape[1]
                    batch['pred_audio'] = F.pad(batch['s1'], (0, T - batch['s1'].shape[1]))
                    batch['pred_audio'] = batch['pred_audio'][:, :T]
                    batch['pred_audio'] = batch['pred_audio']
                    assert batch['pred_audio'].shape == batch['target_audio'].shape
                pred_audios.append(batch['pred_audio'])
                target_audios.append(batch['target_audio'])
                if args.out_dir is not None:
                    target_path = batch['target_audio_path'][0].rsplit('-')[-2].split('/')[-1]
                    os.makedirs(args.out_dir + f"/target/audio/", exist_ok=True)
                    os.makedirs(args.out_dir + f"/pred/audio/", exist_ok=True)
                    torchaudio.save(args.out_dir + f"/target/audio/{target_path}-target.flac",
                                    batch['target_audio'].cpu(), sample_rate=16_000, format='flac')
                    torchaudio.save(args.out_dir + f"/pred/audio/{target_path}-pred.flac",
                                    batch['pred_audio'].cpu() / batch['pred_audio'].norm().cpu() * 20,
                                    sample_rate=16_000, format='flac')
            for pred, target in tqdm(list(zip(pred_audios, target_audios))):
                metric.update(pred, target)
                metricPESQ.update(pred, target)
            print("SISNR", metric.compute().item())
            print("PESQ", metricPESQ.compute().item())

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-o",
        "--out_dir",
        default=None,
        type=str,
        help="Where to save result outputs",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-w",
        "--window_size",
        default=0,
        type=float,
        help="Whether to use window",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirDataset",
                        "args": {
                            "dir": str(test_data_folder),
                        },
                    }
                ],
            }
        }
        print(config.config['data'])
    if config.config.get("data", {}).get("test", None) is not None:
        arg = 'test'
    elif config.config.get("data", {}).get("test-clean", None) is not None:
        arg = 'test-clean'
    elif config.config.get("data", {}).get("val", None) is not None:
        arg = 'val'
    else:
        raise AssertionError("Should provide test!")
    config["data"][arg]["batch_size"] = args.batch_size
    config["data"][arg]["n_jobs"] = args.jobs
    main(config, args)
