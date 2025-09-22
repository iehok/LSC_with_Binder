import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW

from z_config import Config
from z_data_loader import DataLoader
from z_utils import get_model_class, make_folds, save_best_model, set_target_words

config = Config()


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        score = float(trainer.callback_metrics['dev_mse_loss'])
        if score < pl_module.args.best_score:
            pl_module.args.best_score = score
            save_best_model(pl_module)


class PLModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_class = get_model_class(args.projector_model)
        self.model = self.model_class()
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)

    def forward(self, batch, mode):
        if mode == 'train':
            return self.model(batch, self.device)
        elif mode == 'val':
            return self.model.forward_eval(batch, self.device)

    def training_step(self, batch, batch_idx):
        loss = self(batch, mode='train')
        self.log_dict({'train_loss': loss.item()})
        return loss

    def on_validation_epoch_start(self):
        self.mse_loss_lst = []

    def validation_step(self, batch, batch_idx):
        mse_loss = self(batch, mode='val')
        self.mse_loss_lst.append(mse_loss)

    def on_validation_epoch_end(self):
        dev_mse_loss = sum(self.mse_loss_lst) / len(self.mse_loss_lst)
        self.log_dict({'dev_mse_loss': dev_mse_loss})

    def configure_optimizers(self):
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        return self.optimizer


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for espeon

    pl_module = PLModule(args)
    data_module = DataLoader(args, 'train', 'mapping')
    logger = WandbLogger(name=args.run_name, project=config.PROJECT_NAME) if args.wandb else None

    print(f'Trainable params: {sum(p.numel() for p in pl_module.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in pl_module.parameters())}')

    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[CheckpointCallback()],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        devices=args.gpus,
        enable_checkpointing=False,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        num_sanity_val_steps=0,
        precision=args.precision,
        profiler=args.profiler,
        strategy=args.strategy,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(pl_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int)
    parser.add_argument('--embedding_model', type=str, default='bert-base-uncased')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--projector_model', type=str, default='mlp')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--target_span', type=str, default='1910-2000')
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    args.experiment_dir = os.path.join(config.EXP_DIR, args.run_name)
    Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)
    args.args_path = os.path.join(args.experiment_dir, 'args.yaml')
    if args.val_check_interval > 1.0:
        args.val_check_interval = int(args.val_check_interval)
    args.best_score = 1e9

    binder_words = set_target_words('binder')

    target_words = [
        w for w in binder_words
        if os.path.isfile(os.path.join(config.COHA_CENTROIDS_DIR, f'binder_words/{args.embedding_model}/original/{w}.json')) and (w != 'used')
    ]

    make_folds(target_words, os.path.join(args.experiment_dir, 'words_splitted.jsonl'))

    main(args)
