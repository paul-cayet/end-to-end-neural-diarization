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

import json
from pyannote.audio.core.io import get_torchaudio_info
from pyannote.database import registry
from pyannote.database import get_protocol
from pyannote.database import FileFinder
import yaml
import argparse
from tqdm import tqdm


def simple_load_dataset(
        db_config_path: str,
        protocol_fullname: str
    ):
    """Loads a Pyannote protocol from which we can access train/dev/test data
    Args
    ----
    db_config_path: str
        Path to the 'database' configuration file
    protocol_fullname: str
        Protocol full name, structured as: `DatabaseName.TaskName.ProtocolName`

    Returns
    -------
    dataset: Protocol
        The Pyannote dataset
    """

    registry.load_database(db_config_path)
    preprocessors = {'audio': FileFinder(registry=registry)}

    dataset = get_protocol(protocol_fullname, preprocessors=preprocessors) 
    return dataset


def write_torchaudio_json(path, dataset):

    obj = dict()
    datasets = [dataset.train(), dataset.development()]
    for dataset in datasets:
        for i, file in tqdm(enumerate(dataset)):
            try:
                info = get_torchaudio_info(file)
                obj[file["uri"]] = info.__dict__
            except:
                print(f"We lost {file['uri']}")

    with open(path, 'w') as f:
        json.dump(obj, f, indent=2) 


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default="config.yml")
    args = parser.parse_args()
    config_path = args.config_path

    config = yaml.safe_load(open(config_path, "r"))
    training_config = config["training_config"]
    dataset_config = config["dataset"]
    db_config_path = dataset_config["db_config_path"]
    protocol_path = dataset_config["protocol_full_name"]

    dataset = simple_load_dataset(db_config_path,protocol_path)
    path = training_config["precomputed_torchinfo_path"]
    write_torchaudio_json(path, dataset)