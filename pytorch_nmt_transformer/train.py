import logging
from typing import Callable
from timeit import default_timer as timer

import torch.nn as nn
from torch import device as torch_device
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam

from pytorch_nmt_transformer import config as train_config
from pytorch_nmt_transformer.helper import create_mask


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.CrossEntropyLoss,
        optimizer: Adam,
        collate_fn: Callable,
        device: torch_device,
        pad_index: int,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.collate_fn = collate_fn
        self.device = device
        self.pad_index = pad_index

        self.logger = logging.getLogger(__name__)

    def train_epoch(self, dataset: Dataset) -> float:
        self.model.train()
        losses = 0
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=train_config.BATCH_SIZE,
            collate_fn=self.collate_fn,
        )

        for src, target in train_dataloader:
            src = src.to(self.device)
            target = target.to(self.device)

            target_input = target[:-1, :]

            logits = self.model(src, target_input)

            self.optimizer.zero_grad()
            target_out = target[1:, :]
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1)
            )
            loss.backward()

            self.optimizer.step()
            losses += loss.item()

        return losses / len(list(train_dataloader))

    def evaluate(self, val_dataset: Dataset) -> float:
        self.model.eval()
        losses = 0

        val_dataloader = DataLoader(
            val_dataset, batch_size=train_config.BATCH_SIZE, collate_fn=self.collate_fn
        )

        for src, target in val_dataloader:
            src = src.to(self.device)
            target = target.to(self.device)

            target_input = target[:-1, :]

            logits = self.model(src, target_input)

            target_out = target[1:, :]
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1)
            )
            losses += loss.item()

        return losses / len(list(val_dataloader))

    def train(
        self,
        num_epochs: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        for epoch in range(1, num_epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(dataset=train_dataset)
            end_time = timer()

            val_loss = self.evaluate(val_dataset=val_dataset)
            self.logger.info(
                (
                    f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                    f"Epoch time={(end_time - start_time):.3f}s"
                )
            )
