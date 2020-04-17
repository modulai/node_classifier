import os
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from nodepl import NodePlClassifier
import torch
from optuna.structs import TrialState

MODEL_DIR = "models/"
PERCENT_TEST_EXAMPLES = 1
EPOCHS = 20
N_GPUS = 4

class PyTorchLightningPruningCallback(EarlyStopping):
    """PyTorch Lightning callback to prune unpromising trials.
    Example:
        Add a pruning callback which observes validation accuracy.
        .. code::
            trainer.pytorch_lightning.Trainer(
                early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='avg_val_acc'))
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(PyTorchLightningPruningCallback, self).__init__(patience=3,
                                                              mode='max',
                                                              monitor='auc')

        optuna.integration.pytorch_lightning._check_pytorch_lightning_availability()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, trainer, module):
        # Check Early stopping criteria first
        stop_training = super().on_epoch_end(trainer, module)

        current_score = module.metrics.get(self._monitor)
        epoch = module.current_epoch
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.exceptions.TrialPruned(message)
        return stop_training


def find_free_gpu(trial):
    """Find free GPU by using user attributes"""
    if trial.number < N_GPUS:
        gpu = trial.number % N_GPUS
    else:
        trials = study.get_trials()
        trial_gpus = [t.user_attrs.get('gpu') for t in trials if t.state == TrialState.RUNNING and t.number != trial.number]
        # Find first free gpu
        gpu = (set(range(N_GPUS)) - set(trial_gpus)).pop()
    trial.set_user_attr('gpu', gpu)
    return gpu


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)),
        monitor="auc"
    )
    data_file = "training_data.pkl"

    logger = TensorBoardLogger("lightning_logs", name=f"node_classifier_{trial.number}")

    torch.cuda.empty_cache()

    gpu = find_free_gpu(trial)

    trainer = pl.Trainer(
        logger=logger,
        val_percent_check=PERCENT_TEST_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        distributed_backend='gp',
        gpus=[gpu] if torch.cuda.is_available() else None,
        early_stop_callback=PyTorchLightningPruningCallback(trial,
                                                            monitor="auc"),
    )

    batch_size = 64 * trial.suggest_int("batch_size", 2, 4)
    num_trees = 50 * trial.suggest_int("num_trees", 1, 20)
    depth = trial.suggest_int("depth", 3, 8)

    # determine max number of layers not to get OOM errors
    size = 4 * batch_size * num_trees * depth * 2 ** depth/10**9
    max_layers = min(10, int(2/size))

    num_layers = trial.suggest_int("num_layers", 1, max_layers)

    model = NodePlClassifier(data_file,
                             in_features=57,
                             train_fraction=1,
                             num_trees=num_trees,
                             batch_size=batch_size,
                             depth=depth,
                             num_layers=num_layers,
                             lr=0.001,
                             tree_dim=1,
                             gpu=gpu)
    trainer.fit(model)

    return model.metrics['auc']


pruner = optuna.pruners.SuccessiveHalvingPruner()
sampler = optuna.samplers.TPESampler()

study = optuna.create_study(direction="maximize",
                            pruner=pruner,
                            sampler=sampler,
                            study_name='NodePLClassifier',
                            storage="sqlite:///experiment.db",
                            load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=4)

print(study.best_trial)
