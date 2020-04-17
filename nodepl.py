import lib
import pickle
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer


class NodePlClassifier(pl.LightningModule):
    def __init__(self, data_file, train_fraction, batch_size, in_features,
                 num_trees, num_layers, tree_dim, depth,
                 lr, choice='entmax', gpu=0):
        super(NodePlClassifier, self).__init__()
        self.data_file = data_file
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else "cpu"
        choice_function, bin_function = (lib.entmax15, lib.entmoid15) if choice == "entmax" \
            else (lib.sparsemax, lib.sparsemoid)
        self.dense1 = lib.DenseBlock(input_dim=in_features, layer_dim=num_trees,
                                     num_layers=num_layers, tree_dim=tree_dim,
                                     depth=depth, flatten_output=True,
                                     choice_function=choice_function,
                                     bin_function=bin_function).to(self.device)
        self.transf = QuantileTransformer()

        self.metrics = {}

    def prepare_data(self):
        X_train, X_test, self.Y_train, self.Y_test =\
            pickle.load(open(self.data_file, "rb"))

        self.X_train = self.transf.fit_transform(X_train)
        self.X_test = self.transf.transform(X_test)
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.dense1(self.to_torch_tensor(self.X_train[:1000]).to(self.device))

    def to_torch_tensor(self, x):
        return torch.tensor(x.astype(np.float32))

    def train_dataloader(self):

        train_subset = np.random.choice(range(self.X_train.shape[0]),
                                        size=int(self.train_fraction * self.X_train.shape[0]),
                                        replace=False)

        X_train_tensor = self.to_torch_tensor(self.X_train[train_subset])
        Y_train_tensor = self.to_torch_tensor(self.Y_train.values[train_subset])

        return DataLoader(torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor),
                          batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):

        X_test_tensor = self.to_torch_tensor(self.X_test)
        Y_test_tensor = self.to_torch_tensor(self.Y_test.values)

        return DataLoader(torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor),
                          batch_size=self.batch_size, shuffle=False)

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, logits, actual):
        return F.binary_cross_entropy_with_logits(logits, actual)

    def forward(self, x):
        p = self.dense1(x)
        return p.mean(dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return OrderedDict({'loss': loss, 'logits': logits, 'actuals': y,
                            'log': {'val_batch_loss': loss.mean()}})

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs])
        actuals = torch.cat([x['actuals'] for x in outputs])
        loss = torch.tensor([x['loss'] for x in outputs])
        auc = roc_auc_score(actuals.cpu(), logits.cpu())
        self.metrics = {'val_loss': loss.mean(), 'auc': auc}
        return OrderedDict({'val_loss': auc, 'log': self.metrics})


# logger = TensorBoardLogger("lightning_logs", name="node_classifier")

# trainer = pl.Trainer(max_epochs=100, gpus=1, logger=logger)

# # Question: Why should tree_dim be something else than 1 in one-class case?

# net = NodePlClassifier(data_file, train_fraction=1,
#                             batch_size=256, in_features=57,
#                             num_trees=256, num_layers=5, depth=6,
#                             tree_dim=1, lr=0.001)

# trainer.fit(net)
