#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
ML/Data Processing Configuration Modules
----------------------------------------------------------------
This package provides comprehensive configurations and utility
modules for machine learning, NLP, CV, and general data processing.

Main Categories:
+ cfg_base       : Basic file paths and general config (CONFIG, FilePaths)
+ cfg_dl         : Deep learning base parameters (CONFIG4DL, DataPreprocessor, Hyperparameters)
+ cfg_cnn        : CNN-specific parameters (CONFIG4CNN, CNNParams)
+ cfg_mlp        : MLP-specific parameters (CONFIG4MLP, MLPParams)
+ cfg_rnn        : RNN-specific parameters (CONFIG4RNN, RNNParams)
+ cfg_unet       : UNet-specific parameters (CONFIG4UNET, UNetParams)
+ cfg_types      : Type definitions and enums (Attentions, Langs, Tasks, Tokens, SeqStrategies, etc.)
+ parser         : Parser module for command line arguments

Usage:
+ Access default configuration: e.g., CONFIG4CNN.CNN_PARAMS.OUT_CHANNELS
+ Create new instances for custom configurations:
    - from src.configs import Configuration4CNN, CNNParams
    - my_cnn_config = Configuration4CNN(CNN_PARAMS=CNNParams(OUT_CHANNELS=128))
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.3.0"

from .cfg_base import BASE_CONFIG, BaseConfig, Database, FilePaths, Punctuations
from .cfg_cnn import CNN_CONFIG, CNNConfiguration, CNNParams
from .cfg_dl import DL_CONFIG, DLConfiguration, DataPreprocessor, Hyperparameters
from .cfg_enums import (AttnHeads, AttnScorer,
                        Languages,
                        SeqMergeMethods, SeqNets, SeqStrategies,
                        Tasks,
                        SpecialTokens,
                        SeqSeparator)
from .cfg_hf import HF_CONFIG, HuggingFaceConfiguration, HFParams
from .cfg_mlp import MLP_CONFIG, MLPConfiguration, MLPParams
from .cfg_rnn import RNN_CONFIG, RNNConfiguration, RNNParams
from .cfg_seq2seq_tf import S2STF_CONFIG, S2STransformerConfiguration, TransformerParams
from .cfg_unet import UNET_CONFIG, UNetConfiguration, UNetParams
from .parser import set_argument_parser

__all__ = [
    "BASE_CONFIG", "BaseConfig", "Database", "FilePaths", "Punctuations",

    "DL_CONFIG", "DLConfiguration", "DataPreprocessor", "Hyperparameters",

    "CNN_CONFIG", "CNNConfiguration", "CNNParams",

    "AttnHeads", "AttnScorer",
    "Languages",
    "SeqMergeMethods", "SeqNets", "SeqStrategies",
    "Tasks",
    "SpecialTokens",
    "SeqSeparator",

    "HF_CONFIG", "HuggingFaceConfiguration", "HFParams",

    "MLP_CONFIG", "MLPConfiguration", "MLPParams",

    "RNN_CONFIG", "RNNConfiguration", "RNNParams",

    "S2STF_CONFIG", "S2STransformerConfiguration", "TransformerParams",

    "UNET_CONFIG", "UNetConfiguration", "UNetParams",

    "set_argument_parser"
]
