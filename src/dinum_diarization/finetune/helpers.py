# Module imports
from dinum_diarization.utils import (
    Torchaudiopreprocessor,
    generate_unique_logpath,
    _verify_args,
)

# Pyannote imports
from pyannote.database import FileFinder, registry, get_protocol
from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation

# Typing imports
from pyannote.database.protocol import Protocol
from pyannote.audio.core.task import Task
from typing import Callable, List, Dict, Optional
from types import MethodType

# Pytorch imports
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    Callback,
)
from lightning.pytorch.loggers import WandbLogger, Logger
from pytorch_lightning import Trainer
from torch.optim import Adam, SGD
import torch

# Misc imports
import os
import yaml

def load_dataset(
    db_config_path: str,
    protocol_fullname: str,
    torchaudio_path: Optional[str] = None,
) -> Protocol:
    """Loads a Pyannote protocol from which we can access train/dev/test data
    Args
    ----
    db_config_path: str
        Path to the 'database' configuration file
    protocol_fullname: str
        Protocol full name, structured as: `DatabaseName.TaskName.ProtocolName`
    torchaudio_path: Optional[str]
        Path to precomputed Audio MetaData. Default value is None

    Returns
    -------
    dataset: Protocol
        The Pyannote dataset
    """

    registry.load_database(db_config_path)

    preprocessors = {"audio": FileFinder(registry=registry)}
    if torchaudio_path is not None:
        preprocessors["torchaudio.info"] = Torchaudiopreprocessor(torchaudio_path)

    dataset = get_protocol(protocol_fullname, preprocessors=preprocessors)
    element = next((dataset.train()))
    print(dir(element))
    print('first training element: ', [x for x in element.keys()])
    return dataset


def get_dataset(dataset_config: Dict) -> Protocol:

    db_config_path = dataset_config["db_config_path"]
    protocol_fullname = dataset_config["protocol_fullname"]
    torchaudio_path = None
    if "torchaudio_path" in dataset_config:
        torchaudio_path = dataset_config["torchaudio_path"]

    dataset = load_dataset(
        db_config_path=db_config_path,
        protocol_fullname=protocol_fullname,
        torchaudio_path=torchaudio_path,
    )
    return dataset


def load_callbacks(
    task: Task,
    checkpoint_dirname: str,
    save_last: bool = False,
    save_top_k: int = 1,
    every_n_epochs: int = 1,
    save_weights_only: bool = False,
    early_stopping_params: Optional[Dict] = None,
    use_rich_progress_bar: Optional[bool] = None,
    raw_run_name: str = "Syncnet",
) -> List[Callback]:
    """Generates the list of callbacks to be used by the trainer
    Args
    ----
    early_stopping_params: Dict
        Parameters for the early stopping callback. If None, will not be used.
        Default is None

    Returns
    -------
    callbacks: List[Callback]
        The list of callbacks
    """

    monitor, direction = task.val_monitor
    checkpoint_dirname = generate_unique_logpath(checkpoint_dirname, raw_run_name=raw_run_name)
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
        save_last=save_last,
        save_weights_only=save_weights_only,
        filename=f"{monitor}" + "-{epoch:02d}-{val_loss:.2f}",
        verbose=False,
        dirpath=checkpoint_dirname,
    )

    callbacks = [checkpoint]
    if early_stopping_params is not None:
        early_stopping_params = _verify_args(early_stopping_params, ["min_delta", "patience"])

        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=direction,
            strict=True,
            verbose=False,
            **early_stopping_params,
        )
        callbacks.append(early_stopping)

    if use_rich_progress_bar:
        callbacks.append(RichProgressBar())

    return callbacks


def get_callbacks(task: Task, callbacks_config: Dict) -> List[Callback]:

    callback_args = [
        "checkpoint_dirname",
        "save_last",
        "save_top_k",
        "every_n_epochs",
        "save_weights_only",
        "early_stopping_params",
        "use_rich_progress_bar",
    ]
    callbacks_config = _verify_args(callbacks_config, callback_args)

    callbacks = load_callbacks(task=task, **callbacks_config)
    return callbacks


def load_model(
    checkpoint: str,
    use_auth_token: Optional[str] = None,
    device=None,
) -> Model:
    """Loads a pretrained model
    Args
    ----
    checkpoint: str
        Path to checkpoint, or a remote URL, or a model identifier from
        the huggingface.co model hub.
    use_auth_token : str
        When loading a private huggingface.co model, set use_auth_token
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running huggingface-cli login.

    Returns
    -------
    model : Model
        Model
    """
    model = Model.from_pretrained(checkpoint, use_auth_token=use_auth_token, strict=False)
    if device is not None:
        model = model.to(device)
    return model


def _get_device(model_config: Dict):
    device = model_config.get("device", "auto")
    assert device in [
        "gpu",
        "cpu",
        "auto",
    ], f"Device must be in ['gpu','cpu','auto'], is {device}"
    if device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_config: Dict) -> Model:
    checkpoint = model_config["checkpoint"]
    device = _get_device(model_config)
    use_auth_token = os.getenv("HF_AUTH_TOKEN")

    model = load_model(checkpoint=checkpoint, use_auth_token=use_auth_token, device=device)
    return model


def get_segmentation_task(model: Model, dataset: Protocol, task_config: Dict) -> Task:

    task_args = [
        "max_speakers_per_chunk",
        "max_speakers_per_frame",
        "batch_size",
        "num_workers",
        "vad_loss",
    ]
    task_config = _verify_args(task_config, task_args)

    task = Segmentation(
        dataset,
        duration=model.specifications.duration,
        max_num_speakers=len(model.specifications.classes),
        **task_config,
    )
    return task


def load_trainer(
    accelerator: str,
    callbacks: List[Callback],
    max_epochs: int,
    gradient_clip_val: float = 0.5,
    wandb_logger: Optional[Logger] = None,
) -> Trainer:
    """Loads a Pytorch Lightning Trainer
    Args
    ----
    accelerator: str
        accelerator: Supports passing different accelerator
        types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto").
    callbacks: List[Callback]
        logger:
    max_epochs: int
        Maxmimum number of epochs
    gradient_clip_val: float
        Value for the gradient clipping. Defaults to 0.5
    wandb_logger: Logger (or iterable collection of loggers) for experiment tracking.

    Returns
    -------
    trainer : Trainer
        Trainer
    """
    trainer = Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        logger=wandb_logger,
    )
    return trainer


def get_trainer(
    task: Task,
    config: Dict,
    wandb_logger: Optional[Logger] = None,
):

    callbacks_config = config["callbacks"]
    trainer_config = config["trainer"]

    callbacks = get_callbacks(task, callbacks_config)
    trainer = load_trainer(callbacks=callbacks, wandb_logger=wandb_logger, **trainer_config)

    return trainer


def prepare_model_for_training(
    model: Model,
    task: Task,
    optimizer: Callable,
    stage: str = "fit",
):
    """Prepares the model for training by specifying the Task and optimizer
    Args
    ----
    model: Model
        model to be prepared
    stage: str
        Default to `fit` for training

    Returns
    -------
    model : Model
        Prepared model
    """

    model.task = task
    model.prepare_data()
    model.setup(stage=stage)

    model.configure_optimizers = MethodType(optimizer, model)

    return model


def Adam_optimizer(model, lr):
    return Adam(model.parameters(), lr=lr)


def SGD_optimizer(model, lr):
    return SGD(model.parameters(), lr=lr)


def get_optimizer_method(training_config) -> Callable:

    optimizer_method = training_config["optimizer_method"]
    learning_rate = training_config["learning_rate"]

    if optimizer_method == "Adam":
        return lambda model: Adam_optimizer(model, lr=learning_rate)
    elif optimizer_method == "SGD":
        return lambda model: SGD_optimizer(model, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer method {optimizer_method} is not supported yet")


def get_logger(config):
    training_config = config["training_config"]
    wandb_logger = None
    if "wandb_config" in config:
        wandb_config = config["wandb_config"]
        wandb_logger = WandbLogger(
            log_model="all",
            project=wandb_config["project"],
            entity=wandb_config["entity"],
        )
        # On veut ajouter des info de notre config dans le wandb log
        wandb_logger.log_hyperparams(training_config)
    return wandb_logger

