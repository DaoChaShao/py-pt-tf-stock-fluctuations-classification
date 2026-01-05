#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/5 12:47
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_seq2seq_hf.py
# @Desc     :   

from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn, save, load
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from typing import Literal, final


class BaseHFPretrainedTransformer(PreTrainedModel, ABC):
    """ Base Pretrained Transformer Network in the Huggingface """

    def __init__(self, model_name: str, config: PretrainedConfig) -> None:
        """ Constructor """
        super().__init__(config)
        self._model: str = model_name
        self.config: PretrainedConfig = config

        # Initialize the transformer model
        self._transformer = AutoModel.from_pretrained(self._model)
        # Initialize the dropout layer
        self._dropper = nn.Dropout(self.config.hidden_dropout_prob)

    @abstractmethod
    def forward(self, input_ids, attention_mask, token_type_ids):
        """ Forward Pass """
        pass

    @property
    def transformer(self):
        """ Get the transformer layer """
        return self._transformer

    @property
    def dropper(self):
        """ Get the dropout layer """
        return self._dropper

    @abstractmethod
    def summary(self):
        """ Summary the net details """
        pass


if __name__ == "__main__":
    pass
