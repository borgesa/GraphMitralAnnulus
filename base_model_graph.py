#!/usr/bin/env python3
import gc
import os
import sys
import time
from pathlib import Path

import click
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from loguru import logger

from mvq_graph_model.utils.data_and_loss import (
    average_error,
    coordinate_loss,
    curve_to_curve,
    focal_loss,
    max_error,
    perimeter,
    smooth_loss,
)
from mvq_graph_model.visualize.error_plots import plot_hm_dist_error


def initialize_loggers(config, logdir):
    logdir = Path(logdir)
    logdir.mkdir(parents=True)

    logger.remove()
    for sink in config:
        if sink["sink"] == "sys.stderr":
            sink["sink"] = sys.stderr
        elif sink["sink"] == "sys.stdout":
            sink["sink"] = sys.stdout
        else:
            sink["sink"] = logdir / sink["sink"]
        logger.add(**sink)


class Model(LightningModule):
    config: Box
    _log_config: dict = {
        "prog_bar": True,
        "on_step": True,
        "on_epoch": True,
    }

    def __init__(self, config: str | Path | Box, *args, **kwargs):
        super().__init__()

        if isinstance(config, Box):
            self.config = config
        else:
            self.config = Box.from_yaml(
                filename=config, default_box=True, box_dots=True
            )

        # Model:
        self.model = instantiate(self.config.model)

        self.training_augmentation_transform = [
            instantiate(x) for x in self.config.training.augmentation
        ]

        # Validation batches with extra logging (updated by callback):
        self.random_batch_indices = [0]
        self.loss_annulus = coordinate_loss
        logger.info("init finished")
        self.test_results = []
        self.inference_benchmark = False

    def log_losses(self, losses, metrics, phase, batch_size):
        _log_config = {"batch_size": batch_size, **self._log_config}

        # log loss
        total_loss = 0
        for v in losses.values():
            total_loss = total_loss + v

        if isinstance(total_loss, torch.Tensor):
            if total_loss.detach().sum() == 0:
                total_loss += 1e-7
        elif total_loss == 0:
            total_loss = 1e-7

        abs_loss = {k: v.detach() for k, v in losses.items()}
        abs_loss["total"] = total_loss.detach()

        self.logger.experiment.add_scalars(
            f"loss_abs/{phase}",
            {k.split("/")[-1]: v.detach() for k, v in abs_loss.items()},
            global_step=self.global_step,
        )

        self.logger.experiment.add_scalars(
            f"loss_rel/{phase}",
            {k.split("/")[-1]: (v / total_loss).detach() for k, v in losses.items()},
            global_step=self.global_step,
        )

        self.log(f"loss/{phase}", total_loss.detach(), **_log_config)
        self.log(
            f"loss/{phase}_annulus", losses["loss_gcn_local"].detach(), **_log_config
        )

        # log metrics

        tags = list({x.split("/", 1)[0] for x in metrics.keys()})
        for tag in tags:
            keys = [
                key.split("/", 1)[-1] for key in metrics.keys() if key.startswith(tag)
            ]
            metrics_group = {key: metrics[f"{tag}/{key}"].detach() for key in keys}
            self.logger.experiment.add_scalars(
                f"{tag}/{phase}",
                metrics_group,
                global_step=self.global_step,
            )

        return {**losses, "loss": total_loss}

    @logger.catch(reraise=True)
    def forward(self, x):
        return self.model(x)

    @logger.catch(reraise=True)
    def configure_optimizers(self):
        result_dict = {}

        optimizer = instantiate(self.config.training.optimizer, self.model.parameters())
        result_dict["optimizer"] = optimizer
        return result_dict

    @logger.catch(reraise=True)
    def training_step(self, batch, batch_idx):
        phase = "train"
        for transform in self.training_augmentation_transform:
            batch = transform(batch)
        return self.step(batch, batch_idx, phase)

    @logger.catch(reraise=True)
    def validation_step(self, batch, batch_idx):
        phase = "val"
        return self.step(batch, batch_idx, phase)

    @logger.catch(reraise=True)
    def test_step(self, batch, batch_idx):
        metrics = {}

        volume = batch["volume"]
        annulus_coords = batch["target"]
        dmap = batch["dmap"]
        dmap_ao = batch["dmap_ao"]
        dmap_weights = batch["dmap_weights"]
        dmap_ao_weights = batch["dmap_ao_weights"]

        predictions = self.model(volume)

        if self.inference_benchmark:
            return

        patient_id = batch["patient"].cpu().item()

        batch_size = volume.shape[0]

        if batch_size != 1:
            raise RuntimeError(
                f"Test metrics assume batch of 1!!! Batch size is: {batch_size}"
            )

        edge_size = volume.shape[-1]

        P_gcn_local = predictions["gcn_prediction"]
        P_gcn_global = predictions["gcn_prediction_global"]

        # ~ plot_test_results(volume, predictions, dmap, annulus_coords,P_gcn_global,P_gcn_local)

        c2c_global = curve_to_curve(
            P_gcn_global,
            annulus_coords,
            interpolation_points=10,
            scaling=self.config.meta.normalized_to_mm,
        ).mean()

        c2c_out = curve_to_curve(
            P_gcn_local,
            annulus_coords,
            interpolation_points=10,
            scaling=self.config.meta.normalized_to_mm,
        ).mean()

        metrics["c2c_annulus/c2c_global"] = c2c_global
        metrics["c2c_annulus/c2c_local"] = c2c_out

        p1 = perimeter(P_gcn_global)
        l1 = perimeter(annulus_coords)
        metrics["rel_perimeter_global"] = (torch.abs(p1 - l1) / l1).ravel() * 100

        p1 = perimeter(P_gcn_local)
        l1 = perimeter(annulus_coords)
        metrics["rel_perimeter_local"] = (torch.abs(p1 - l1) / l1).ravel() * 100

        metrics = {k: v.cpu().numpy() for k, v in metrics.items()}
        self.test_results.append((patient_id, metrics))

    def on_test_epoch_end(self, *args, **kwargs):
        keys = {x[0] for x in self.test_results}
        tensor_names = []
        for x in self.test_results:
            tensor_names.extend(x[1].keys())

        tensor_names = tuple(sorted(set(tensor_names)))

        stats = {k: {} for k in tensor_names}
        for key in keys:
            for k, v in stats.items():
                v[key] = []

        for x in self.test_results:
            patientid = x[0]
            for k, v in x[1].items():
                stats[k][patientid].append(v)

        for tname in tensor_names:
            logger.info("--------------------------------------------------------")
            logger.info(f"Statistics for {tname}:")
            mean = np.mean([np.mean(v) for k, v in stats[tname].items()])
            std = np.std([np.mean(v) for k, v in stats[tname].items()])
            median = np.median([np.mean(v) for k, v in stats[tname].items()])
            abs_worse = np.max([np.max(v) for k, v in stats[tname].items()])
            logger.info(f"Mean +/-std for mean-patient: {mean:1.4} +/- {std:1.4}")
            logger.info(f"Median for mean-patient: {median:1.4}")
            logger.info(f"Absolute worst without weighting: {abs_worse:1.4}")

        # clear up statistics, just in case somebody by accident runs stats twice
        self.test_results = []

    @logger.catch(reraise=True)
    def step(self, batch, batch_idx, phase):

        losses = {}
        metrics = {}

        if phase == "train":
            total_steps = len(self.trainer.datamodule.train_dataloader())
            relative_steps = (batch_idx + 1) / total_steps
        else:
            relative_steps = 1.0

        volume = batch["volume"]
        annulus_coords = batch["target"]
        dmap = batch["dmap"]
        dmap_ao = batch["dmap_ao"]
        dmap_weights = batch["dmap_weights"]
        dmap_ao_weights = batch["dmap_ao_weights"]

        predictions = self.model(volume)

        batch_size = volume.shape[0]
        edge_size = volume.shape[-1]

        th = self.config.meta.metrics_threshold

        P_gcn_global = predictions["gcn_prediction_global"]
        P_gcn_local = predictions["gcn_prediction"]
        P_cnn = predictions["cnn_prediction"][:, 0:1, ...]
        P_cnn_aorta = predictions["cnn_prediction"][:, 1:2, ...]

        losses["loss_gcn_global"] = self.loss_annulus(P_gcn_global, annulus_coords)
        losses["loss_gcn_local"] = self.loss_annulus(P_gcn_local, annulus_coords)
        losses["loss_cnn"] = smooth_loss(P_cnn, dmap, dmap_weights)
        losses["loss_cnn_aorta"] = smooth_loss(P_cnn_aorta, dmap_ao, dmap_ao_weights)

        for k, v in losses.items():
            losses[k] = losses[k] * self.config.loss.weights[k]
        losses = focal_loss(losses, self.config.loss.focal_gamma)

        # warmup for losses
        for k, v in losses.items():
            warmup_epoch = self.config.loss.first_epoch.get(k, 0)
            warmup_length = self.config.loss.warm_up_length.get(k, 1)

            scale = 0
            if warmup_epoch > self.current_epoch:
                scale = 0
            elif warmup_epoch + warmup_length <= self.current_epoch:
                scale = 1
            elif (warmup_epoch <= self.current_epoch) and (
                warmup_epoch + warmup_length > self.current_epoch
            ):
                scale = (
                    self.current_epoch + relative_steps - warmup_epoch
                ) / warmup_length

            scale = np.power(scale, self.config.loss.get("warm_up_scaling_gamma", 2.0))

            self.logger.experiment.add_scalars(
                f"relative_lr/{phase}",
                {k: scale},
                global_step=self.global_step,
            )

            losses[k] = losses[k] * self.config.loss.weights[k] * scale

        losses = focal_loss(losses, self.config.loss.focal_gamma)

        if "annulus_baseline" in self.config.meta:
            metrics["annulus/baseline"] = torch.tensor(
                [self.config.meta.annulus_baseline], dtype=torch.float32
            )
            metrics["c2c_annulus/baseline"] = metrics["annulus/baseline"]

        metrics["annulus/cnn"] = (
            average_error(P_cnn, dmap, threshold=th) * self.config.meta.pixel_to_mm
        )
        metrics["annulus/cnn_max"] = (
            max_error(P_cnn, dmap, threshold=th) * self.config.meta.pixel_to_mm
        )
        metrics["annulus/cnn_fullV"] = (
            average_error(P_cnn, dmap) * self.config.meta.pixel_to_mm
        )
        metrics["annulus/gcn_global"] = (
            torch.linalg.norm(P_gcn_global - annulus_coords, dim=-1, ord=2).mean()
            * edge_size
        ) * self.config.meta.pixel_to_mm
        metrics["annulus/gcn_local"] = (
            torch.linalg.norm(P_gcn_local - annulus_coords, dim=-1, ord=2).mean()
            * edge_size
        ) * self.config.meta.pixel_to_mm

        # Curve to curve metrics + log (to allow callback to see):
        c2c_global_nip = curve_to_curve(
            P_gcn_global, annulus_coords, scaling=self.config.meta.normalized_to_mm
        ).mean()
        c2c_out_nip = curve_to_curve(
            P_gcn_local, annulus_coords, scaling=self.config.meta.normalized_to_mm
        ).mean()

        c2c_global = curve_to_curve(
            P_gcn_global,
            annulus_coords,
            interpolation_points=10,
            scaling=self.config.meta.normalized_to_mm,
        ).mean()

        c2c_out = curve_to_curve(
            P_gcn_local,
            annulus_coords,
            interpolation_points=10,
            scaling=self.config.meta.normalized_to_mm,
        ).mean()

        metrics["c2c_annulus/c2c_global_nip"] = c2c_global_nip
        metrics["c2c_annulus/c2c_local_nip"] = c2c_out_nip
        metrics["c2c_annulus/c2c_global"] = c2c_global
        metrics["c2c_annulus/c2c_local"] = c2c_out

        c2c_global_loss = curve_to_curve(
            P_gcn_global,
            annulus_coords,
            interpolation_points=10,
            gamma=self.config.loss.c2c_loss.get("gamma", 1.0),
            scaling=1,  # Keeping metrics in [0, 1]
        ).mean()

        c2c_out_loss = curve_to_curve(
            P_gcn_local,
            annulus_coords,
            interpolation_points=10,
            gamma=self.config.loss.c2c_loss.get("gamma", 1.0),
            scaling=1,
        ).mean()

        losses["loss_c2c_global"] = c2c_global_loss  # also acts as a loss
        losses["loss_c2c_local"] = c2c_out_loss

        self.log(
            f"metric/{phase}_c2c_out",
            c2c_out.detach(),
            batch_size=batch_size,
            **self._log_config,
        )
        self.log(
            f"metric/{phase}_c2c_global",
            c2c_global.detach(),
            batch_size=batch_size,
            **self._log_config,
        )

        metrics["aorta/cnn_aorta"] = (
            average_error(P_cnn_aorta, dmap_ao, threshold=th)
            * self.config.meta.pixel_to_mm
        )
        metrics["aorta/cnn_aorta_max"] = (
            max_error(P_cnn_aorta, dmap_ao, threshold=th) * self.config.meta.pixel_to_mm
        )
        metrics["aorta/cnn_aorta_fullV"] = (
            average_error(P_cnn_aorta, dmap_ao) * self.config.meta.pixel_to_mm
        )

        losses = self.log_losses(losses, metrics, phase=phase, batch_size=batch_size)

        if (
            phase == "train" and batch_idx % self.config.meta.plotting_frequency == 0
        ) or (phase == "val" and batch_idx in self.random_batch_indices):
            fig = plot_hm_dist_error(volume, dmap, P_cnn, landmarks=2)
            self.logger.experiment.add_figure(
                f"{phase}/cnn_annulus", fig, self.global_step
            )

            fig = plot_hm_dist_error(
                volume,
                dmap_ao,
                P_cnn_aorta,
                landmarks=1,
            )
            self.logger.experiment.add_figure(
                f"{phase}/cnn_aorta", fig, self.global_step
            )

            for key, v in predictions.items():
                if "us_features" in key:
                    for dim in range(v.shape[1]):
                        fig = plt.figure()
                        ax = fig.gca()
                        ax.imshow(
                            v[0, dim, :, 64, :].detach().cpu().numpy().T, origin="lower"
                        )
                        ax.set_title(f"{phase}/{key}_ch{dim}")

                        self.logger.experiment.add_figure(
                            f"{phase}/{key}_ch{dim}", fig, self.global_step
                        )
                        fig = None
                        gc.collect()

            self.logger.experiment.flush()
            fig = None
            plt.close("all")

        gc.collect()
        return {**losses, **metrics}


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--slurm_job_id", default="")
@logger.catch(reraise=True)
def main(config, slurm_job_id):
    logger.info(f"Load config file: {config}")
    config = Box.from_yaml(filename=config, default_box=True, box_dots=True)

    if slurm_job_id:
        config.meta.slurm_id = slurm_job_id

    if config.meta.working_dir == "." or not config.meta.working_dir:
        working_dir = Path.cwd()
    else:
        os.chdir(config.meta.working_dir)
        working_dir = config.meta.working_dir

    if config.meta.seed:
        seed = config.meta.seed
    else:
        # Random seed
        seed = int(time.time())
        config.meta.seed = seed
    config_to_file = config.to_yaml()

    pl.seed_everything(seed)
    config.to_yaml()

    torch.set_float32_matmul_precision(config.meta.float32_matmul_precision)

    trainer = instantiate(config.trainer)

    logger.info("Initialize loggers")
    initialize_loggers(config.logger, trainer.logger.log_dir)
    logger.info(f"Working directory: {working_dir}")
    (Path(trainer.logger.log_dir) / "config.yaml").write_text(config_to_file)

    model = Model(config)
    datamodule = instantiate(config.datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
