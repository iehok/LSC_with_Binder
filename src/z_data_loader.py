import pytorch_lightning as pl
from torch.utils.data import DataLoader as dl

from src.z_datasets import DevData, TestData, TrainData


class DataLoader(pl.LightningDataModule):
    def __init__(self, args, mode, task):
        super().__init__()
        self.args = args
        self.mode = mode
        self.task = task

    def setup(self, stage):
        if self.mode == 'train':
            self.train_data = TrainData(self.args)
            self.dev_data = DevData(self.args, self.task)
        elif self.mode == 'val':
            self.dev_data = DevData(self.args, self.task)
        elif self.mode == 'test':
            self.test_data = TestData(self.args, self.task)

    def train_dataloader(self):
        return dl(
            batch_size=self.args.batch_size,
            collate_fn=self.train_data.collate_fn,
            dataset=self.train_data,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        return dl(
            batch_size=1,
            collate_fn=self.dev_data.collate_fn,
            dataset=self.dev_data,
            num_workers=0,
            shuffle=False,
        )

    def test_dataloader(self):
        return dl(
            batch_size=1,
            collate_fn=self.test_data.collate_fn,
            dataset=self.test_data,
            num_workers=0,
            shuffle=False,
        )
