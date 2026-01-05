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
                 batch_size: int | Literal[8, 16, 32, 64] = 32, shuffle_state: bool = True,
                 workers: int | Literal[0, 4] = 0,
                 drop_last: bool = False
                 ) -> None:
        """ Initialise the HFDataLoaderForClassification class
        :param dataset: the Hugging Face Dataset to load data from
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

        self._iterator: Iterator | None = None

    @staticmethod
    def _collate_fn(batch):
        """ Batch collation function
        :param batch: List of samples
        :return: Batched features and labels
        """
        result = {}

        # Gather all keys in the dataset
        for key in batch[0].keys():
            values = [item[key] for item in batch]

            # Convert to Tensor and stack if necessary
            if isinstance(values[0], Tensor):
                result[key] = stack(values)
            else:
                result[key] = tensor(values)

        return result

    def __repr__(self) -> str:
        """ String representation of the DataLoader
        :return: String representation
        """
        return (f"HFDataLoaderForClassification("
                f"dataset_size={len(self._ds)}, "
                f"columns={self.columns}, "
                f"batch_size={self.batch_size}, "
                f"num_batches={len(self)}, "
                f"num_workers={self.num_workers}, "
                f"drop_last={self.drop_last})")

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
    def columns(self) -> list[str]:
        """ Get the columns of the dataset
        :return: List of column names
        """
        return self._ds.column_names


if __name__ == "__main__":
    pass
