#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/5 15:44
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_trainer.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from torch import nn, device
from torch.utils.data import DataLoader
from typing import Literal, Protocol

from src.utils.PT import get_device

WIDTH: int = 64


class BaseTorchTrainerProtocol(Protocol):
    """ Protocol for Base Torch Trainer """
    losses: Signal

    def _epoch_train(self, dataloader: DataLoader) -> float: ...

    def _epoch_valid(self, dataloader: DataLoader) -> tuple[float, dict]: ...

    def fit(self): ...


class BaseTorchTrainer(QObject):
    """ Base torch trainer class for managing training process """
    losses: Signal = Signal(int, float, float)

    def __init__(self,
                 model: nn.Module, optimiser, criterion,
                 accelerator: str | Literal["auto", "cuda", "cpu"] = "auto",
                 scheduler=None
                 ) -> None:
        """ Initialize the Trainer
        :param model: PyTorch model to be trained
        :param optimiser: Optimiser for training
        :param criterion: Loss function
        :param accelerator: Device to place the model on
        :param scheduler: Learning rate scheduler
        """
        super().__init__()
        self._accelerator = get_device(accelerator)
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler

    def _epoch_train(self, dataloader: DataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        pass

    def _epoch_valid(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        pass

    def fit(self,
            train_loader: DataLoader, valid_loader: DataLoader, epochs: int,
            model_save_path: str | None = None,
            log_name: str | None = None,
            patience: int = 5
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :param log_name: name for the log file
        :param patience: number of epochs to wait for improvement before early stopping
        :return: None
        """
        pass

    @property
    def model(self) -> nn.Module:
        """ Get the model """
        return self._model

    @property
    def optimiser(self):
        """ Get the optimiser """
        return self._optimiser

    @property
    def criterion(self):
        """ Get the loss criterion """
        return self._criterion

    @property
    def scheduler(self):
        """ Get the learning rate scheduler """
        return self._scheduler

    @property
    def accelerator(self) -> str:
        """ Get the accelerator device """
        return self._accelerator


if __name__ == "__main__":
    pass
