#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/5 16:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_hf_trainer.py
# @Desc     :   

from datetime import datetime
from json import dumps
from torch import nn, optim, device, no_grad, save
from torch.utils.data import DataLoader
from typing import Literal, override

from src.trainers.base_trainer import BaseTorchTrainer
from src.trainers.calc_classification import calculator_for_classification
from src.trainers.calc_cm import calculator_for_confusion_metrics
from src.utils.logger import record_log

WIDTH: int = 64


class Seq2SeqHFTransformerTrainer(BaseTorchTrainer):
    """ Trainer for Seq2Seq Huggingface Transformer Models """

    def __init__(self,
                 model: nn.Module, optimiser, criterion,
                 accelerator: str | Literal["auto", "cuda", "cpu"] = "auto",
                 scheduler=None,
                 clip_grad: bool = False,
                 ) -> None:
        """ Initialize the Trainer
        :param model: PyTorch model to be trained
        :param optimiser: Optimiser for training
        :param criterion: Loss function
        :param accelerator: Device to place the model on
        :param scheduler: Learning rate scheduler
        :param clip_grad: Gradient clipping value
        """
        super().__init__(
            model=model,
            optimiser=optimiser,
            criterion=criterion,
            accelerator=accelerator,
            scheduler=scheduler
        )
        self._clip: bool = clip_grad

    @override
    def _epoch_train(self, dataloader: DataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for batch in dataloader:
            batch = {k: v.to(device(self._accelerator)) for k, v in batch.items()}
            labels = batch.pop("PriceVariation")

            if labels.dim() > 1:
                labels = labels.squeeze()

            self._optimiser.zero_grad()
            outputs = self._model(**batch)
            # print(outputs.shape, labels.shape)

            loss = self._criterion(outputs, labels)
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0) if self._clip else None
            loss.backward()

            self._optimiser.step()

            _loss += loss.item() * next(iter(batch.values())).size(0)
            _total += next(iter(batch.values())).size(0)

        return _loss / _total

    @override
    def _epoch_valid(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _total: float = 0.0

        _results: list[int] = []
        _targets: list[int] = []
        with no_grad():
            for batch in dataloader:
                batch = {k: v.to(device(self._accelerator)) for k, v in batch.items()}
                labels = batch.pop("PriceVariation")

                outputs = self._model(**batch)
                # print(outputs.shape, labels.shape)

                if labels.dim() > 1:
                    labels = labels.squeeze()
                # print(outputs.shape, labels.shape)

                loss = self._criterion(outputs, labels)
                _loss += loss.item() * next(iter(batch.values())).size(0)
                _total += next(iter(batch.values())).size(0)

                _results.extend(outputs.argmax(dim=1).cpu().tolist())
                _targets.extend(labels.cpu().tolist())

        _metrics: dict[str, float] = {
            **calculator_for_classification(_results, _targets),
            **calculator_for_confusion_metrics(_results, _targets),
        }

        return _loss / _total, _metrics

    @override
    def fit(self,
            train_loader: DataLoader, valid_loader: DataLoader, epochs: int,
            *,
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
        # Initialize logger
        timer = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        logger = record_log(f"train_at_{timer}-{log_name}")

        _best_valid_loss = float("inf")
        _min_delta = 5e-4
        _patience_counter = 0
        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, _metrics = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss)

            # Log epoch results
            dps: int = 4
            logger.info(dumps({
                "epochs": epochs,
                "epoch": epoch + 1,
                "alpha": self._optimiser.param_groups[0]["lr"],
                "train_loss": round(float(train_loss), dps),
                "valid_loss": round(float(valid_loss), dps),
                **_metrics,
            }))

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss - _min_delta:
                _patience_counter = 0
                _best_valid_loss = valid_loss

                if model_save_path:
                    save(self._model.state_dict(), model_save_path)
                    print(f"√ Model's parameters saved to {model_save_path}\n")
            else:
                _patience_counter += 1
                print(f"× No improvement [{_patience_counter}/{patience}]\n")
                if _patience_counter >= patience:
                    print("*" * WIDTH)
                    print("Early Stopping Triggered")
                    print("-" * WIDTH)
                    print(f"Early stopping at epoch {epoch}, the best value is {_best_valid_loss:.4f}.")
                    print("*" * WIDTH)
                    print()
                    break

            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(valid_loss)
                else:
                    self._scheduler.step()

        if _patience_counter < patience:
            print(f"Training completed after {epochs} epochs.")


if __name__ == "__main__":
    pass
