import os
import argparse
import collections
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
import src.utils.audio as audio_module
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")
    # torch.autograd.set_detect_anomaly(True)
    text_encoder = None # config.get_text_encoder()
    # setup data_loader instances
    audio = config.init_obj(config['audio'], audio_module)
    dataloaders = get_dataloaders(config, audio)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    print(device, device_ids)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    if config['trainer'].get('compile', False):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            loss_module = torch.compile(loss_module, mode='reduce-overhead')
        except Exception:
            print("Could not compile loss")
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer
    )

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        audio=audio,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_step=config["trainer"].get("log_step", 200),
        log_predictions_step_epoch=config["trainer"].get(
            "log_predictions_step_epoch", 1
        ),
        mixed_precision=config["trainer"].get("mixed_precision", True),
    )

    trainer.train()


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
        "-r",
        "--resume",
        default=None,
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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
