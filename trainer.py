#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/2 22:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from pathlib import Path
from torch import optim, nn
from transformers import AutoConfig

from src.configs.cfg_hf import HF_CONFIG
from src.configs.parser import set_argument_parser
from src.trainers.seq2seq_hf_trainer import Seq2SeqHFTransformerTrainer
from src.nets.seq2seq_hf import Seq2SeqHFTransformer
from src.utils.PT import TorchRandomSeed

from pipeline.preparer import prepare_dataloader


def main() -> None:
    """ Main Function """
    # Set up argument parser
    args = set_argument_parser()

    with TorchRandomSeed("Chinese to English (Seq2Seq) Translation", tick_tock=True):
        # Get the data
        train, valid = prepare_dataloader()

        # Initialize model
        net: Path = Path(HF_CONFIG.FILE_PATHS.PRETRAINED_MODEL)
        print(net)
        if net.exists():
            print(f"Model {net.name} already exists")
            config = AutoConfig.from_pretrained(str(net))
        else:
            print("Downloading pretrained model first")
            config = AutoConfig.from_pretrained(HF_CONFIG.PARAMETERS.NET_CN)

        model = Seq2SeqHFTransformer(config.name_or_path, config, num_labels=2)
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=HF_CONFIG.HYPERPARAMETERS.DECAY)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        criterion = nn.CrossEntropyLoss()
        model.summary()
        """
        ****************************************************************
        Model Summary for Seq2SeqHFTransformer
        ----------------------------------------------------------------
        Vocab Size:                21128
        Hidden Size:               768
        Number of Layers:          12
        Number of Labels:          2
        Number of Attention Heads: 12
        Dropout Rate:              0.1
        ----------------------------------------------------------------
        Total parameters:          102,269,186
        Trainable parameters:      102,269,186
        Non-trainable parameters:  0
        ****************************************************************
        """

        # Setup trainer
        trainer = Seq2SeqHFTransformerTrainer(
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            accelerator=HF_CONFIG.HYPERPARAMETERS.ACCELERATOR,
            scheduler=scheduler,
            clip_grad=True,
        )
        # Train the model
        trainer.fit(
            train_loader=train,
            valid_loader=valid,
            epochs=args.epochs,
            model_save_path=str(HF_CONFIG.FILE_PATHS.SAVED_NET),
            log_name=f"{HF_CONFIG.PARAMETERS.NET_CN.split("/")[1]}",
            patience=5,
        )
        """
        ****************************************************************
        Training Summary:
        ----------------------------------------------------------------
        "epoch": 8/100, "train_loss": 0.2203, "valid_loss": 0.2748, "accuracy": 0.8872, "precision": 0.8878, "recall": 0.8872, "f1_score": 0.8872
        ****************************************************************
        """


if __name__ == "__main__":
    main()
