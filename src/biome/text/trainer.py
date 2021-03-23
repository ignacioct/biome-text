import logging
import os
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
from allennlp.common import Params
from allennlp.data import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training.optimizers import Optimizer
from pytorch_lightning import Callback
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.profiler import BaseProfiler
from torch.utils.data import IterableDataset

from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstanceDataset
from biome.text.pipeline import Pipeline

# We do not require wandb
_HAS_WANDB = False
try:
    import wandb
except ImportError:
    pass
else:
    wandb.ensure_configured()
    _HAS_WANDB = True


_LOGGER = logging.getLogger(__name__)


class Trainer:
    """A class for training a `biome.text.Pipeline`.

    It is basically a light wrapper around the awesome Pytorch Lightning Trainer to facilitate the interaction
    with our pipelines. The docs are mainly a copy from the
    [Lightning Trainer API](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer)
    with some additional parameters added.

    Parameters
    ----------
    accelerator
        Previously known as distributed_backend (dp, ddp, ddp2, etc...).
        Can also take in an accelerator object for custom hardware.

    accumulate_grad_batches
        Accumulates grads every k batches or as set up in the dict.

    add_csv_logger
        Adds a default CSV logger if `logger` is not False. Default: True

    add_tensorboard_logger
        Adds a default Tensorboard logger if `logger` is not False. Default: True

    add_wandb_logger
        Adds a default WandB logger if `logger` is not False and wandb is installed. Default: True

    amp_backend
        The mixed precision backend to use ("native" or "apex")

    amp_level
        The optimization level to use (O1, O2, etc...).

    auto_lr_find
        If set to True, will make trainer.tune() run a learning rate finder,
        trying to optimize initial learning for faster convergence. trainer.tune() method will
        set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
        To use a different key set a string instead of True with the key name.

    auto_scale_batch_size
        If set to True, will `initially` run a batch size
        finder trying to find the largest batch size that fits into memory.
        The result will be stored in self.batch_size in the LightningModule.
        Additionally, can be set to either `power` that estimates the batch size through
        a power search or `binsearch` that estimates the batch size through a binary search.

    auto_select_gpus
        If enabled and `gpus` is an integer, pick available
        gpus automatically. This is especially useful when
        GPUs are configured to be in "exclusive mode", such
        that only one process at a time can access them.

    batch_size
        Size of the batch.

    benchmark
        If true enables cudnn.benchmark.

    callbacks
        Add a callback or list of callbacks.

    checkpoint_callback
        If ``True``, enable checkpointing.
        It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. Default: ``True``.

    check_val_every_n_epoch
        Check val every n train epochs.

    data_bucketing
        If enabled, try to apply data bucketing over training batches.

    default_root_dir
        Default path for logs and weights when no logger/ckpt_callback passed.
        Can be remote file paths such as 's3://mybucket/path' or 'hdfs://path/'
        Default: './training_logs'.

    deterministic
        If true enables cudnn.deterministic.

    fast_dev_run
        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
        of train, val and test to find any bugs (ie: a sort of unit test).

    flush_logs_every_n_steps
        How often to flush logs to disk (defaults to every 100 steps).

    gpus
        number of gpus to train on (int) or which GPUs to train on (list or str) applied per node

    gradient_clip_val
        0 means don't clip.

    limit_train_batches
        How much of training dataset to check (floats = percent, int = num_batches)

    limit_val_batches
        How much of validation dataset to check (floats = percent, int = num_batches)

    limit_test_batches
        How much of test dataset to check (floats = percent, int = num_batches)

    log_every_n_steps
        How often to log within steps (defaults to every 50 steps).

    log_gpu_memory
        None, 'min_max', 'all'. Might slow performance

    logger
        Logger (or iterable collection of loggers) for experiment tracking.
        If not False, we will add some loggers by default, see `add_[csv, tensorboard, wandb]_logger`.
        Default: True

    prepare_data_per_node
        If True, each LOCAL_RANK=0 will call prepare data.
        Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

    process_position
        orders the progress bar when running multiple models on same machine.

    progress_bar_refresh_rate
        How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
        Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
        a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

    profiler
        To profile individual steps during training and assist in identifying bottlenecks. Passing bool
        value is deprecated in v1.1 and will be removed in v1.3.

    overfit_batches
        Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0

    plugins
        Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.

    precision
        Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.

    max_epochs
        Stop training once this number of epochs is reached. Disabled by default (None).
        If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.

    min_epochs
        Force training for at least these many epochs. Disabled by default (None).
        If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.

    max_steps
        Stop training after this number of steps. Disabled by default (None).

    min_steps
        Force training for at least these number of steps. Disabled by default (None).

    num_nodes
        number of GPU nodes for distributed training.

    num_processes
        number of processes for distributed training with distributed_backend="ddp_cpu"

    num_sanity_val_steps
        Sanity check runs n validation batches before starting the training routine.
        Set it to `-1` to run all batches in all validation dataloaders. Default: 2

    optimizer
        Configuration for an [AllenNLP/PyTorch optimizer](https://docs.allennlp.org/main/api/training/optimizers/)
        that is constructed via the AllenNLP configuration framework.
        Default: `{"type": "adam", "lr": 0.001}`

    reload_dataloaders_every_epoch
        Set to True to reload dataloaders every epoch.

    replace_sampler_ddp
        Explicitly enables or disables sampler replacement. If not specified this
        will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
        train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
        you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

    resume_from_checkpoint
        Path/URL of the checkpoint from which training is resumed. If there is
        no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
        training will start from the beginning of the next epoch.

    sync_batchnorm
        Synchronize batch norm layers between process groups/whole world.

    terminate_on_nan
        If set to True, will terminate training (by raising a `ValueError`) at the
        end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

    tpu_cores
        How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

    track_grad_norm
        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

    truncated_bptt_steps
        Truncated back prop breaks performs backprop every k steps of much longer
        sequence.

    val_check_interval
        How often to check the validation set. Use float to check within a training epoch,
        use int to check every n steps (batches).

    weights_summary
        Prints a summary of the weights when training begins.

    weights_save_path
        Where to save weights if specified. Will override default_root_dir
        for checkpoints only. Use this if for whatever reason you need the checkpoints
        stored in a different place than the logs written in `default_root_dir`.
        Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
        Defaults to `default_root_dir`.

    move_metrics_to_cpu
        Whether to force internal logged metrics to be moved to cpu.
        This can save some gpu memory, but can make training slower. Use with attention.

    multiple_trainloader_mode
        How to loop over the datasets when there are multiple train loaders.
        In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
        and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
        reload when reaching the minimum length of datasets.

    stochastic_weight_avg
        Whether to use `Stochastic Weight Averaging (SWA)
        <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_`
    """

    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = "top",
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, bool, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[Plugin, str, list]] = None,
        amp_backend: str = "native",
        amp_level: str = "O2",
        distributed_backend: Optional[str] = None,
        automatic_optimization: Optional[bool] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        stochastic_weight_avg: bool = False,
        # non lightning trainer parameters,
        add_csv_logger: bool = True,
        add_tensorboard_logger: bool = True,
        add_wandb_logger: bool = True,
        batch_size: int = 16,
        data_bucketing: bool = False,
        optimizer: Optional[Dict] = None,
    ):
        default_root_dir = default_root_dir or os.path.join(
            os.getcwd(), "training_logs"
        )
        self._lightning_trainer_kwargs = {
            key: value
            for key, value in locals().items()
            # filter non Lightening Trainer kwargs
            if key
            not in [
                "self",
                "add_csv_logger",
                "add_tensorboard_logger",
                "add_wandb_logger",
                "batch_size",
                "data_bucketing",
                "optimizer",
            ]
        }
        # these configurations will be logged to wandb
        self._trainer_config = {
            "batch_size": batch_size,
            "data_bucketing": data_bucketing,
            "optimizer": optimizer or {"type": "adam", "lr": 0.001},
        }

        self._add_csv_logger = add_csv_logger
        self._add_tensorboard_logger = add_tensorboard_logger
        self._add_wandb_logger = add_wandb_logger

        # the wandb logger holds a special place in our heart
        self._wandb_logger: Optional[WandbLogger] = None

        if logger is not False:
            self._lightning_trainer_kwargs["logger"] = self._add_default_loggers()

        self.trainer = pl.Trainer(**self._lightning_trainer_kwargs)

    def _add_default_loggers(self) -> List[LightningLoggerBase]:
        """Adds default loggers for the lightning trainer"""
        loggers = self._lightning_trainer_kwargs["logger"]
        if loggers is True:
            loggers = []
        elif isinstance(loggers, LightningLoggerBase):
            loggers = [loggers]

        def loggers_include(logger_type) -> bool:
            return any([isinstance(logger, logger_type) for logger in loggers])

        if self._add_csv_logger and not loggers_include(CSVLogger):
            loggers.append(
                CSVLogger(
                    save_dir=self._lightning_trainer_kwargs["default_root_dir"]
                    or os.getcwd(),
                    name="csv",
                )
            )
        if self._add_tensorboard_logger and not loggers_include(TensorBoardLogger):
            loggers.append(
                TensorBoardLogger(
                    save_dir=self._lightning_trainer_kwargs["default_root_dir"]
                    or os.getcwd(),
                    name="tensorboard",
                )
            )
        if self._add_wandb_logger and _HAS_WANDB and not loggers_include(WandbLogger):
            save_dir = self._lightning_trainer_kwargs["default_root_dir"] or os.getcwd()
            self._wandb_logger = WandbLogger(save_dir=save_dir, project="biome")
            loggers.append(self._wandb_logger)
        elif loggers_include(WandbLogger):
            self._wandb_logger = [
                logger for logger in loggers if isinstance(logger, WandbLogger)
            ][0]
        # somehow the wandb dir does not get created, i think this is a bug on pl side, have to check it out
        if self._wandb_logger is not None and not os.path.isdir(
            os.path.join(self._wandb_logger.save_dir, "wandb")
        ):
            os.makedirs(os.path.join(self._wandb_logger.save_dir, "wandb"))

        return loggers

    def fit(
        self,
        pipeline: Pipeline,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        include_valid_data_in_vocab: bool = False,
        lazy: bool = False,
    ):
        """Train the pipeline

        Parameters
        ----------
        pipeline
            Pipeline to train
        train_dataset
            The training dataset
        valid_dataset
            The validation dataset
        vocab_config
            A `VocabularyConfiguration` to create/extend the pipeline's vocabulary.
            If 'default' (str), we will use the default configuration `VocabularyConfiguration()`.
            If None, we will leave the pipeline's vocabulary untouched. Default: 'default'.
        include_valid_data_in_vocab
            If True, take the validation data into account when creating the vocabulary (apart from the training data).
            Has no effect if `vocab_config` is None. Default: False.
        lazy
            If True, instances are lazily loaded from disk, otherwise they are loaded into memory. Default: False.
        """
        # create instances
        train_instances = train_dataset.to_instances(pipeline, lazy=lazy)
        valid_instances = (
            None
            if valid_dataset is None
            else valid_dataset.to_instances(pipeline, lazy=lazy)
        )

        # create vocab
        vocab_config = (
            VocabularyConfiguration() if vocab_config == "default" else vocab_config
        )
        if vocab_config is not None:
            vocab_datasets = [train_instances]
            if valid_instances is not None and include_valid_data_in_vocab:
                vocab_datasets += [valid_instances]
            pipeline.create_vocab(vocab_datasets, config=vocab_config)

        # create dataloaders
        train_dataloader = create_dataloader(
            train_instances,
            batch_size=self._trainer_config["batch_size"],
            data_bucketing=self._trainer_config["data_bucketing"],
        )
        valid_dataloader = (
            create_dataloader(
                valid_dataset.to_instances(pipeline, lazy=lazy),
                batch_size=self._trainer_config["batch_size"],
                data_bucketing=self._trainer_config["data_bucketing"],
            )
            if valid_dataset is not None
            else None
        )

        # create optimizer
        pipeline.model.optimizer = Optimizer.from_params(
            Params(
                {
                    "model_parameters": pipeline.model.named_parameters(),
                    **self._trainer_config["optimizer"],
                }
            )
        )

        # log config to wandb
        if self._wandb_logger is not None:
            config = {
                "pipeline": pipeline.config.as_dict(),
                "trainer": self._trainer_config,
            }
            self._wandb_logger.experiment.config.update(config)

        self.trainer.fit(
            pipeline.model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


def create_dataloader(
    instance_dataset: InstanceDataset,
    batch_size: int = 16,
    data_bucketing: bool = False,
) -> PyTorchDataLoader:
    """Returns a pytorch DataLoader for AllenNLP instances

    Parameters
    ----------
    instance_dataset
        The dataset of instances for the DataLoader
    batch_size
        Batch size
    data_bucketing
        If True, tries to sort batches with respect to the maximum input lengths per batch.
        Not supported for lazily loaded data!

    Returns
    -------
    data_loader
    """
    if data_bucketing and isinstance(instance_dataset, IterableDataset):
        _LOGGER.warning(
            "'data_bucketing' is not supported for lazily loaded data. We will deactivate it."
        )
        data_bucketing = False

    return PyTorchDataLoader(
        instance_dataset,
        batch_size=1 if data_bucketing else batch_size,
        batch_sampler=BucketBatchSampler(
            data_source=instance_dataset,
            batch_size=batch_size,
        )
        if data_bucketing
        else None,
    )
