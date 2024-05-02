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

from pyannote.audio import __version__
from pyannote.audio import Pipeline
import argparse
import torch
from tqdm import tqdm
import os
import lightning as pl
import glob
import dotenv
from pathlib import Path
import yaml


def save_model(
        pipeline: Pipeline,
        model_saving_path: str
):
    """Function to save the model
    Args
    ----
    pipeline: Pipeline
        The pipeline object
    model_saving_path: str
        Path to save the model

    Returns
    -------
    None
    """
    model = pipeline._segmentation.model
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint["pytorch-lightning_version"] = pl.__version__
    torch.save(checkpoint, model_saving_path)

    print(f"Successfully saved the model at {model_saving_path}")

    return None

def get_pipeline_config(
        pipeline: Pipeline
):
    '''Function to get the pipeline configuration
    Args
    ----
    pipeline: Pipeline
        The pipeline object

    Returns
    -------
    config: dict
        The configuration of the pipeline
    '''
    config = {
        "version":  __version__,
        "pipeline": {
            "name" : '.'.join(pipeline.__module__.split('.')[:-1] + [pipeline.__class__.__name__]),
            "params" : {
                "clustering" : pipeline.klustering,
                "embedding" : pipeline.embedding,
                "embedding_batch_size" : pipeline.embedding_batch_size,
                "embedding_exclude_overlap" : pipeline.embedding_exclude_overlap,
                # Our ckpt :
                # "segmentation" : model_saving_path,
                "segmentation_batch_size" : pipeline.segmentation_batch_size
            }
        },
        "params": {
            "clustering": {
                "method": pipeline.clustering._instantiated["method"],
                "min_cluster_size": pipeline.clustering._instantiated["min_cluster_size"],
                "threshold": pipeline.clustering._instantiated["threshold"]
            },
            "segmentation": {
                "min_duration_off": pipeline.segmentation._instantiated["min_duration_off"]
            }
        }
    }

    return config

def load_model(
        model_saving_path: str,
        pipeline: Pipeline
):
    """Function to load the model
    Args
    ----
    model_saving_path: str
        Path to the saved model
    pipeline: Pipeline
        The pipeline object

    Returns
    -------
    pipeline: Pipeline
        The pipeline object with the loaded model
    """
    config = {
        "version":  __version__,
        "pipeline": {
            "name" : '.'.join(pipeline.__module__.split('.')[:-1] + [pipeline.__class__.__name__]),
            "params" : {
                "clustering" : pipeline.klustering,
                "embedding" : pipeline.embedding,
                "embedding_batch_size" : pipeline.embedding_batch_size,
                "embedding_exclude_overlap" : pipeline.embedding_exclude_overlap,
                # Our ckpt :
                "segmentation" : model_saving_path,
                "segmentation_batch_size" : pipeline.segmentation_batch_size
            }
        },
        "params": {
            "clustering": {
                "method": pipeline.clustering._instantiated["method"],
                "min_cluster_size": pipeline.clustering._instantiated["min_cluster_size"],
                "threshold": pipeline.clustering._instantiated["threshold"]
            },
            "segmentation": {
                "min_duration_off": pipeline.segmentation._instantiated["min_duration_off"]
            }
        }
    }

    print(f"Configuration of the pipeline: {config}")

    # Save it as temp.yaml
    with open("temp.yaml", "w") as f:
        yaml.dump(config, f)
    pipeline = Pipeline.from_pretrained(checkpoint_path="temp.yaml")

    print(f"Successfully loaded the model from {model_saving_path}")

    # delete
    os.remove("temp.yaml")

    return pipeline

def predict(
        dataset: str, 
        audio_path: str,
        n: int = None,
        ckpt: str = None
        ):
    """Function to predict the speaker diarization on the audio files
    Args
    ----
    dataset: str
        Name of the dataset
    audio_path: str
        Path to the audio files
    n: int
        Number of files to predict

    Returns
    -------
    None
    """
    dotenv.load_dotenv(dotenv.find_dotenv())
    token = os.getenv("HF_TOKEN")

    use_cuda = torch.cuda.is_available()
    assert use_cuda, "CUDA not available"
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
        ).to(device)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pipeline = load_model(ckpt, pipeline)
        pipeline = pipeline.to(device)
        print("Model loaded")
    
    audio_paths = glob.glob(os.path.join(audio_path, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_paths)} audio files")
    pred_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "predictions")
    pred_folder = os.path.join(pred_folder, dataset)
    os.makedirs(pred_folder, exist_ok=True)


    temp_path = ".temp_" + dataset + "/"
    os.makedirs(temp_path, exist_ok=True)

    count = 0
    for file_name in tqdm(audio_paths, desc='Processing files'):

        short_name = file_name.split('/')[-1][:-4]
        prediction_path = os.path.join(pred_folder, short_name + '.rttm')

        print(f'\nTemp path: {temp_path}')
        print(f'Filename: {file_name}')

        if os.path.exists(prediction_path) or os.path.exists(os.path.join(temp_path, '.' + short_name + '.rttm')):
            print('Already predicted/predicting ' + short_name + ', skipping...')
            continue
        else:
            print('Predicting ' + short_name + '...')
            with open(os.path.join(temp_path, '.' + short_name + '.rttm'), "w") as f:
                f.write("")
            count += 1

        diarization = pipeline(file_name)

        with open(prediction_path, "w") as f:
            diarization.write_rttm(f)

        os.remove(os.path.join(temp_path, '.' + short_name + '.rttm'))

        if n is not None and count >= n:
            print(f"Predicted {n} files, stopping\n")
            break

    # Instead, find all the .temp* dir, if one is empty, remove it : 
    temp_dirs = glob.glob('.temp*')
    for temp_dir in temp_dirs:
        if len(os.listdir(temp_dir)) == 0:
            print(f"Removing {temp_dir}")
            os.rmdir(temp_dir)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which dataset to predict")
    parser.add_argument("--data", type=str, help="Dataset to predict")
    parser.add_argument("--audio", type=str, help="Audio files to predict")
    parser.add_argument("--ckpt", type=str, help="Path to the segmentation model checkpoint (optionnal)")
    parser.add_argument("--n", type=int, help="Number of files to predict")

    args = parser.parse_args()

    assert args.data, "No dataset provided (--data)"
    assert args.audio, "No audio path provided (--audio)"
    assert os.path.exists(args.audio), "Audio path not found"

    if args.n is None:
        print("Doing the inference on all files")
    else:
        print(f"Doing the inference on the first {'file' if args.n == 1 else str(args.n) + ' files'}")


    predict(args.data, args.audio, args.n, args.ckpt)