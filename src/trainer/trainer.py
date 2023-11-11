import random
from pathlib import Path
from random import shuffle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import PIL
import pandas as pd
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.metric import PESQ, SiSDR
from src.metric.utils import si_sdr
from src.utils import inf_loop, MetricTracker
from torch.cuda.amp import GradScaler
from src.utils import optional_autocast


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        audio,
        dataloaders,
        text_encoder,
        log_step=400,  # how often WANDB will log
        log_predictions_step_epoch=5,
        mixed_precision=False,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model, criterion, metrics, optimizer, config, device, lr_scheduler
        )
        self.skip_oom = skip_oom
        self.audio = audio
        self.train_dataloader = dataloaders["train"]
        self.embeds_batches_to_log = config["trainer"].get("embeds_batches_to_log", 4)
        self.emb_vis = config["trainer"].get("emb_vis", "pca")
        self.text_encoder = text_encoder
        self.config = config
        self.accumulation_steps = config["trainer"].get("accumulation_steps", 1)
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.log_predictions_step_epoch = log_predictions_step_epoch
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            "loss",
            "ce_loss",
            "snr_loss",
            "grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            "loss",
            "ce_loss",
            "snr_loss",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.pesq = PESQ(16_000, "wb")
        self.scaler = GradScaler(enabled=self.mixed_precision)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_for_gpu in [
            "target_audio",
            "mix_audio",
            "mix_audio_len",
            "reference_audio",
            "reference_audio_len",
            "target_audio_len",
            "speaker_target",
            "target_spectrogram",
            "mix_spectrogram",
            "reference_spectrogram",
            "anchor",
            "positive",
            "negative",
            "mix_phase",
            "target_phase",
            "reference_phase",
        ]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    batch_idx=batch_idx,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("loss", batch["loss"].detach().cpu().item())
            if not batch_idx % self.accumulation_steps:
                self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar(
                    "epoch",
                    epoch,
                )
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), self.train_metrics.avg("loss")
                    )
                )
                self.writer.add_scalar(
                    "learning rate",
                    self.optimizer.state_dict()["param_groups"][0]["lr"],
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if epoch % self.log_predictions_step_epoch == 0:
                    if "anchor" not in batch:
                        self._log_predictions(**batch, is_train=True)
                    else:
                        self._log_embeddings(**batch)
                    if "pred_spectrogram" in batch:
                        self._log_spectrogram(batch)
                    if "pred_phase" in batch:
                        self._log_spectrogram(batch, phase=True)
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        if self.lr_scheduler is not None:
            metric = self.evaluation_metrics.avg("loss")
            self.lr_scheduler.step(metrics=metric)
        return log

    def process_batch(
        self, batch, batch_idx: int, is_train: bool, metrics: MetricTracker
    ):
        if is_train and not batch_idx % self.accumulation_steps:
            self.optimizer.zero_grad()
        batch = self.move_batch_to_device(batch, self.device)
        with optional_autocast(self.mixed_precision):
            outputs = self.model(**batch)
            if type(outputs) is dict:
                batch.update(outputs)
            if self.config["arch"]["type"] == "SpExPlus":
                batch["pred_audio"] = torch.nan_to_num(batch["s1"], 0)
            elif self.config["arch"]["type"].startswith("VoiceFilter"):
                spectrogram = batch.get("pred_spectrogram", batch["mix_spectrogram"])
                phase = batch.get("pred_phase", batch["mix_phase"])
                waveform_reconstructed = torch.stack(
                    [self.audio.spec2wav(s, ph) for s, ph in zip(spectrogram, phase)]
                )
                batch["pred_audio"] = waveform_reconstructed
            criterion = self.criterion(**batch)
            batch.update(criterion)

        if is_train:
            self.scaler.scale(batch["loss"] / self.accumulation_steps).backward()
            if not batch_idx % self.accumulation_steps:
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        metrics.update("loss", batch["loss"].item())
        if "ce_loss" in batch and "snr_loss" in batch:
            metrics.update("ce_loss", batch["ce_loss"].detach().cpu().item())
            metrics.update("snr_loss", batch["snr_loss"].detach().cpu().item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.criterion.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=False,
                        batch_idx=batch_idx,
                        metrics=self.evaluation_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning(
                            "OOM on batch in EVALUATION!!! Skipping batch."
                        )
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)

            if "spectrogram_pred" in batch:
                self._log_spectrogram(batch)
            self._log_predictions(**batch)
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_embeddings(self, **batch):
        if self.writer is None:
            return
        embeddings = []
        labels = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_dataloader):
                anchors = batch["anchor"].cuda()
                anchor_embedding = self.model.make_embedding(anchors)
                embeddings.append(anchor_embedding)
                labels.append(batch["speaker_target"])
                if i > self.embeds_batches_to_log:
                    break
        embeddings = torch.cat(embeddings, dim=0)
        labels = np.concatenate(labels, axis=0)
        if self.emb_vis.lower() == "tsne":
            embeddings = embeddings.cpu().numpy()
            tsne = TSNE(n_components=2, random_state=0)
            embeddings_2d = tsne.fit_transform(embeddings)
            name = "2D TSNE"
        else:
            _, _, Q = torch.pca_lowrank(embeddings, q=2)
            embeddings_2d = torch.mm(embeddings, Q).cpu().numpy()
            name = "2D PCA"
        fig, ax = plt.subplots(figsize=(5, 5))

        for i, label in enumerate(set(labels)):
            idxs = [j for j, x in enumerate(labels) if x == label]
            ax.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label)

        ax.legend()
        self.writer.add_image(name, wandb.Image(fig))
        plt.close()

    @torch.no_grad()
    def _log_predictions(
        self,
        pred_audio,
        target_audio,
        reference_audio,
        mix_audio,
        target_audio_path,
        target_audio_len,
        reference_audio_len,
        mix_audio_len,
        examples_to_log=10,
        is_train=False,
        **kwargs,
    ):
        if self.writer is None:
            return
        tuples = [
            pred_audio,
            target_audio,
            reference_audio,
            mix_audio,
            target_audio_path,
            target_audio_len,
            reference_audio_len,
            mix_audio_len,
        ]
        tuples = [i[:examples_to_log] for i in tuples]
        tuples = list(zip(*tuples))
        sr = self.config["preprocessing"]["sr"]
        shuffle(tuples)
        rows = {}
        for (
            pred_audio_,
            target_audio_,
            reference_audio_,
            mixed_audio_,
            path_,
            target_len,
            ref_len,
            mix_len,
        ) in tuples:
            target_len = target_len.item()
            ref_len = ref_len.item()
            mix_len = mix_len.item()
            pred_audio__ = (
                20 * pred_audio_.detach().type(torch.float32) / pred_audio_.norm()
            )
            pred_audio__ = torch.nn.functional.pad(
                pred_audio__, (0, target_len - pred_audio__.shape[0])
            )
            pred_audio__ = torch.nan_to_num(pred_audio__, 0, 0, 0)
            rows[Path(path_).name] = {
                "reference": wandb.Audio(
                    reference_audio_.squeeze().cpu()[:ref_len], sample_rate=sr
                ),
                "mixed": wandb.Audio(
                    mixed_audio_.squeeze().cpu()[:mix_len], sample_rate=sr
                ),
                "predicted": wandb.Audio(
                    pred_audio__.cpu()[:target_len], sample_rate=sr
                ),
                "target": wandb.Audio(
                    target_audio_.squeeze().cpu()[:target_len], sample_rate=sr
                ),
                "sisdr": si_sdr(pred_audio__[:target_len], target_audio_[:target_len]),
                "pesq": self.pesq(
                    pred_audio__[:target_len], target_audio_[:target_len]
                ),
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    @staticmethod
    def make_image(spectrogram):
        return ToTensor()(PIL.Image.open(plot_spectrogram_to_buf(spectrogram)))

    @torch.no_grad()
    def _log_spectrogram(self, batch, phase=False):
        idx = -1
        name = "phase" if phase else "spectrogram"
        spectrogram_mix = batch[f"mix_{name}"][idx].detach().cpu()
        spectrogram_pred = batch[f"pred_{name}"][idx].detach().cpu()
        spectrogram_target = batch[f"target_{name}"][idx].detach().cpu()
        self.writer.add_image(
            f"{name} target",
            Trainer.make_image(plot_spectrogram_to_buf(spectrogram_target)),
        )
        self.writer.add_image(
            f"{name} pred",
            Trainer.make_image(plot_spectrogram_to_buf(spectrogram_pred)),
        )
        self.writer.add_image(
            f"{name} mix",
            Trainer.make_image(plot_spectrogram_to_buf(spectrogram_mix)),
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        # nan occurs in first batch in first run with grad scaler
                        torch.nan_to_num(p.grad, nan=0).detach(),
                        # p.grad.detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
