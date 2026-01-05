#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/5 13:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_hf.py
# @Desc     :   

from pathlib import Path
from torch import (nn,
                   no_grad, randint, ones, zeros, long)
from transformers import (PretrainedConfig,
                          AutoConfig)
from typing import override, final

from src.configs.cfg_hf import HF_CONFIG

from src.nets.base_seq2seq_hf import BaseHFPretrainedTransformer

WIDTH: int = 64


class Seq2SeqHFTransformer(BaseHFPretrainedTransformer):
    """ A classier using transformer in the huggingface """

    def __init__(self, model_name: str, config: PretrainedConfig, num_labels: int = 1) -> None:
        super().__init__(model_name, config)
        self._classes = num_labels

        # Initialise a classification layer
        self._linear = nn.Linear(self.config.hidden_size, self._classes)

    @override
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = outputs.last_hidden_state[:, 0, :]
        outputs = self.dropper(outputs)
        logits = self._linear(outputs)

        return logits

    @final
    def _count_parameters(self) -> tuple[int, int]:
        """ Count total and trainable parameters
        :return: total and trainable parameters
        """
        total_params = 0
        trainable_params = 0

        for param in self.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return total_params, trainable_params

    @override
    def summary(self):
        """ Print a summary of the model parameters """
        print("*" * WIDTH)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"Vocab Size:                {self.config.vocab_size}")
        print(f"Hidden Size:               {self.config.hidden_size}")
        print(f"Number of Layers:          {self.config.num_hidden_layers}")
        print(f"Number of Labels:          {self.config.num_labels}")
        print(f"Number of Attention Heads: {self.config.num_attention_heads}")
        print(f"Dropout Rate:              {self.config.hidden_dropout_prob}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total parameters:          {total_params:,}")
        print(f"Trainable parameters:      {trainable_params:,}")
        print(f"Non-trainable parameters:  {total_params - trainable_params:,}")
        print("*" * WIDTH)


if __name__ == "__main__":
    net: Path = Path(HF_CONFIG.FILE_PATHS.PRETRAINED_MODEL)
    print(net)
    if net.exists():
        print(f"Model {net.name} already exists")
        config = AutoConfig.from_pretrained(str(net))
    else:
        print("Downloading pretrained model first")
        config = AutoConfig.from_pretrained(HF_CONFIG.PARAMETERS.NET_CN)
    config.num_labels = 2

    model = Seq2SeqHFTransformer(config.name_or_path, config, num_labels=2)
    model.summary()

    with no_grad():
        input_ids = randint(0, config.vocab_size, (2, HF_CONFIG.PROCESSOR.MAX_SEQ_LEN))
        attention_mask = ones((2, HF_CONFIG.PROCESSOR.MAX_SEQ_LEN), dtype=long)
        token_type_ids = zeros((2, HF_CONFIG.PROCESSOR.MAX_SEQ_LEN), dtype=long)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        print("*" * WIDTH)
        print(f"Test forward pass successful as following:")
        print("-" * WIDTH)
        print(f"Output shape: {outputs.shape}")  # Should be (2, num_labels)
        print(f"Output: {outputs}")
        print("*" * WIDTH)
