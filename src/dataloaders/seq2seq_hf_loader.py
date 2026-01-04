#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/4 23:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_hf_loader.py
# @Desc     :   

from datasets import Dataset
from torch import Tensor, tensor, stack
from torch.utils.data import DataLoader
from typing import Literal, Iterator


class HFDataLoaderForClassification(DataLoader):
    """ Hugging Face Data Loader """

    def __init__(self,
                 dataset: Dataset,
                 *,
                 features_col: str, labels_col: str,
                 batch_size: int | Literal[8, 16, 32, 64] = 32, shuffle_state: bool = True,
                 workers: int | Literal[0, 4] = 0,
                 drop_last: bool = False
                 ) -> None:
        """ Initialise the HFDataLoaderForClassification class
        :param dataset: the Hugging Face Dataset to load data from
        :param features_col: the column name for features
        :param labels_col: the column name for labels
        :param batch_size: the number of samples per batch
        :param shuffle_state: whether to shuffle the data at every epoch
        :param workers: the number of workers to use for data loading
        :param drop_last: whether to drop the last incomplete batch
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_state,
            num_workers=workers,
            collate_fn=self._collate_fn,
            pin_memory=False if workers == 0 else True,
            drop_last=drop_last,
        )
        self._ds: Dataset = dataset
        self._features: str = features_col
        self._labels: str = labels_col

        self._iterator: Iterator | None = None

    def _collate_fn(self, batch):
        """ Batch collation function
        :param batch: List of samples
        :return: Batched features and labels
        """
        # Gather features and labels
        features = [item[self._features] for item in batch]
        labels = [item[self._labels] for item in batch]

        # Convert to tensors
        features_tensor = stack(features) if isinstance(features[0], Tensor) else tensor(features)
        labels_tensor = stack(labels) if isinstance(labels[0], Tensor) else tensor(labels)

        return {
            self._features: features_tensor,
            self._labels: labels_tensor
        }

    def _info(self) -> dict:
        """ Get information about the DataLoader
        :return: Information dictionary
        """
        return {
            "dataset_size": len(self._ds),
            "features_column": self._features,
            "labels_column": self._labels,
            "batch_size": self.batch_size,
            "num_batches": len(self),
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
        }

    def __repr__(self) -> str:
        """ String representation of the DataLoader
        :return: String representation
        """
        info = self._info()

        return (f"HFDataLoaderForClassification("
                f"dataset_size={info['dataset_size']}, "
                f"features='{info['features_column']}', "
                f"labels='{info['labels_column']}', "
                f"batch_size={info['batch_size']}, "
                f"num_batches={info['num_batches']}, "
                f"num_workers={info['num_workers']}, "
                f"drop_last={info['drop_last']})")

    def __iter__(self):
        """ Get iterator over the DataLoader
        :return: Iterator
        """
        self._iterator = super().__iter__()

        return self

    def __next__(self):
        """ Get next batch from the DataLoader
        :return: Next batch
        """
        if self._iterator is None:
            self._iterator = super().__iter__()

        return next(self._iterator)

    @property
    def features_column(self) -> str:
        """ Get the features column name
        :return: Features column name
        """
        return self._features

    @property
    def labels_column(self) -> str:
        """ Get the labels column name
        :return: Labels column name
        """
        return self._labels


if __name__ == "__main__":
    pass
