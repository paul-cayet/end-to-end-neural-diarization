# GNU GENERAL PUBLIC LICENSE Version 3

# Copyright (C) 2024 - P. Cayet, N. Ibanez and L. Rondier

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import yaml
from typing import Dict
import dotenv
from dotenv import load_dotenv

from neural_diarization.finetune.helpers import (
    get_dataset,
    get_model,
    get_trainer,
    get_segmentation_task,
    get_logger,
    get_optimizer_method,
    prepare_model_for_training,
    Model,
)

load_dotenv(dotenv.find_dotenv())


def finetune(config: Dict) -> Model:
    """Calls every functions of helpers.py to finetune the model
    Args
    ----
    config: Dict

    Returns
    -------
    model: Model
    """
    logger = get_logger(config)
    dataset = get_dataset(config["dataset"])

    model = get_model(config["model"])
    task = get_segmentation_task(model, dataset, config["task"])
    optimizer = get_optimizer_method(config["training_config"])

    model = prepare_model_for_training(model, task, optimizer)
    trainer = get_trainer(task, config, logger)

    trainer.fit(model)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", default="config.yml")
    args = parser.parse_args()
    config_path = args.config_path

    config = yaml.safe_load(open(config_path, "r"))

    finetune(config)
