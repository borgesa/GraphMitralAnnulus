#!/usr/bin/env python
import sys
from datetime import datetime

import click
import torch
from box import Box
from hydra.utils import instantiate
from loguru import logger

from base_model_graph import Model


@click.option("--checkpoint", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path(exists=True))
@click.option("--dataset", type=click.Path(exists=True))
@click.option("--shape-template", type=click.Path(exists=True))
@click.option("--log", "logfile", type=str, default=None)
@click.option(
    "--benchmark",
    "benchmark",
    is_flag=True,
    default=False,
    help="Enable benchmark mode",
)
@click.command()
def main(checkpoint, config_path, dataset, shape_template, logfile, benchmark):

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if logfile is None:
        dt = datetime.now()
        logfile = f"{dt.year}_{dt.month}_{dt.day}-{dt.hour}:{dt.minute}:{dt.second}.log"
    logger.add(logfile, level="INFO")

    config = Box.from_yaml(filename=config_path, default_box=True, box_dots=True)
    config.trainer.logger.save_dir = "."

    config.model._args_[1].shape_template.path = shape_template
    config.model._args_[2].shape_template.path = shape_template

    model = Model(config)
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    logger.info("Model restored")

    config.datamodule.data_dir = dataset

    datamodule = instantiate(config.datamodule)
    trainer = instantiate(config.trainer)
    if benchmark:
        model.inference_benchmark = True
    trainer.test(model, datamodule=datamodule)


main()
