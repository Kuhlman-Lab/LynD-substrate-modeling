import sys
import wandb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

# import lightning as pl
# from pl.callbacks import ModelCheckpoint
# from py.loggers import CSVLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from torchmetrics.classification import BinaryAccuracy, Accuracy
from torchmetrics import AUROC

from dataset import SequenceDataset
from network import load_model


def clf_metrics():    
    return {
        "accuracy": BinaryAccuracy(), 
        "balanced_accuracy": Accuracy(task='multiclass', num_classes=2, average='macro')
    }


class SequenceModelPL(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, args):
        super().__init__()
        self.model = load_model(args)
        print('Model:\n', self.model, '\n', '=' * 50)
        self.args = args
        
        self.learn_rate = self.args.learning_rate
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            for name, metric in clf_metrics().items():
                self.metrics[split][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        features, targets = batch
        preds = torch.squeeze(self(features), -1)
        loss = F.binary_cross_entropy(preds, targets.to(torch.float32))
        
        for name, metric in self.metrics[f"{prefix}_metrics"].items():
            if name == 'balanced_accuracy':
                metric.update(torch.round(preds), targets)
            else:
                metric.update(preds, targets)

        for name, metric in self.metrics[f"{prefix}_metrics"].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                        batch_size=len(batch))
        if loss == 0.0:
            return None
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        return opt



def train(args):
    print('Starting training!', '\n', '=' * 50)
    print(args, '\n', '=' * 50)
    
    # load dataset and dataloader
    train_dataset = SequenceDataset(args.sele, args.anti, features=args.features, variable_region=args.variable_region, filter_seq=args.filter_seq)
    generator = torch.Generator()
    if args.seed != -1:
        generator = generator.manual_seed(args.seed)
    
    train_dataset, val_dataset, _ = random_split(train_dataset, lengths=(0.8, 0.1, 0.1), generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.cpus, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.cpus, shuffle=False)

    # load lightning model wrapper (loads the model inside it)
    model = SequenceModelPL(args)
        
    # additional params, logging, checkpoints for training
    filename = args.run + '_{epoch:02d}_{val_accuracy:.02}'
    monitor = f'val_accuracy'
        
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = CSVLogger(save_dir='./logs', name=args.run)
        
    # load trainer and fit model
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, 
                         callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, limit_train_batches=args.frac)
    trainer.fit(model, train_loader, val_loader)    
    
    return




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sele', type=str, help='Selection sequence csv', 
                        default='/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/A3_Sup_pos_LynD_R1_001.csv')
    parser.add_argument('--anti', type=str, help='Antiselection sequence csv', 
                        default='/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/A5_Elu_pos_LynD_R1_001.csv')
    parser.add_argument('--run', type=str, help='Run name for logging', default='test')
    parser.add_argument('--cpus', type=int, help='Total CPUs to use for workers', default=8)
    parser.add_argument('--seed', type=int, help='random seed for dataset splits', default=-1)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=2048)
    parser.add_argument('--epochs', type=int, help='training epochs', default=5)
    parser.add_argument('--learning_rate', type=float, help='initial network learning rate', default=1e-3)
    parser.add_argument('--features', type=str, help='which features to use (onehot, continuous, or ECFP)', default='onehot')
    parser.add_argument('--model', type=str, help='Which model type to use (MLP, )', default='MLP')
    parser.add_argument('--frac', type=float, help='batch fraction for training', default=1.0)
    parser.add_argument('--variable_region', nargs='+', type=int, help='list of variable region positions', default=None)
    parser.add_argument('--filter_seq', type=str, default=None, help='sequence filtering criteria (e.g., C1-4 means remove any seqs with Cys in position 1-4)')
    train(parser.parse_args())

