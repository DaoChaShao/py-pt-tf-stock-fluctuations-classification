#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_rnn.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class RNNParams:
    BEAM_SIZE: int = 5
    CLASSES: int = 3  # Binary classification is 2
    EMBEDDING_DIMS: int = 128
    HIDDEN_SIZE: int = 256
    LAYERS: int = 2


@dataclass
class RNNConfiguration:
    DATABASE: Database = field(default_factory=Database)
    FILE_PATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: RNNParams = field(default_factory=RNNParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


RNN_CONFIG: RNNConfiguration = RNNConfiguration()
