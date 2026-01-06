#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prediction.py
# @Desc     :   

from pathlib import Path
from random import randint
from streamlit import (empty, sidebar, subheader, session_state,
                       button, rerun, columns, caption,
                       markdown, write, selectbox)
from torch import load, Tensor, no_grad
from transformers import AutoConfig

from src.configs.cfg_hf import HF_CONFIG
from src.nets.seq2seq_hf import Seq2SeqHFTransformer
from src.utils.HF import load_hf_data_as_ds_dict
from src.utils.helper import Timer
from src.utils.PT import item2tensor

empty_messages: empty = empty()
display4result: empty = empty()
original, prediction = columns(2, gap="medium", vertical_alignment="center", width="stretch")

session4init: list[str] = ["model", "ds_dict", "timer4init"]
for session in session4init:
    session_state.setdefault(session, None)
session4pick: list[str] = ["idx", "sample", "label", "timer4pick"]
for session in session4pick:
    session_state.setdefault(session, None)
session4pred: list[str] = ["prediction", "timer4pred"]
for session in session4pred:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Translater Settings")

    # Load model parameters
    params: Path = Path(HF_CONFIG.FILE_PATHS.SAVED_NET)
    pretrained_params: Path = Path(HF_CONFIG.FILE_PATHS.PRETRAINED_MODEL)

    if params.exists() and pretrained_params.exists():
        empty_messages.warning("The model & dictionary file already exists. You can initialise model first.")

        if session_state["model"] is None:
            # Set a pretrained model
            model: str = selectbox(
                "Select a Model",
                options=[HF_CONFIG.PARAMETERS.NET_CN, ], index=0,
                disabled=True,
                width="stretch"
            )
            caption(f"You selected **{model}** for translation.")

            if button("Initialise Model & Dictionary & Data", type="primary", width="stretch"):
                with Timer("Initialisation") as session_state["timer4init"]:
                    # Initialise a model and load saved parameters
                    config = AutoConfig.from_pretrained(str(pretrained_params))
                    session_state["model"] = Seq2SeqHFTransformer(config.name_or_path, config, num_labels=2)
                    session_state["model"].load_state_dict(
                        load(str(params), map_location=HF_CONFIG.HYPERPARAMETERS.ACCELERATOR)
                    )
                    session_state["model"].to(HF_CONFIG.HYPERPARAMETERS.ACCELERATOR)
                    session_state["model"].eval()
                    print("Model Loaded Successfully!")

                    # Initialise the test data from sqlite database
                    # Load processed dataset dictionary
                    session_state["ds_dict"] = load_hf_data_as_ds_dict(
                        HF_CONFIG.FILE_PATHS.DATASETS,
                        load_train=False, load_valid=False, load_test=True
                    )
                    print(session_state["ds_dict"])
                    rerun()
        else:
            empty_messages.info(f"Initialisation completed! {session_state["timer4init"]} Pick up a data to test.")

            if session_state["idx"] is None and session_state["sample"] is None and session_state["label"] is None:
                if button("Pick up a Data", type="primary", width="stretch"):
                    with Timer("Pick a piece of data") as session_state["timer4pick"]:
                        # Pick up a random sample
                        # Randomly select a data point for prediction
                        session_state["idx"]: int = randint(0, len(session_state["ds_dict"]["test"]) - 1)
                        session_state["sample"]: dict[str, list] = session_state["ds_dict"]["test"][
                            session_state["idx"]
                        ]
                        # print(session_state["sample"])

                        # Convert the token to a tensor
                        session_state["sample"]: dict[str, Tensor] = {
                            key: item2tensor(
                                value, embedding=True, accelerator=HF_CONFIG.HYPERPARAMETERS.ACCELERATOR
                            ).unsqueeze(0)
                            for key, value in session_state["sample"].items()
                        }
                        # print(session_state["sample"])
                        session_state["label"]: Tensor = session_state["sample"].pop("PriceVariation")
                        # print(session_state["sample"])
                        # print(session_state["label"])
                        rerun()

                if button("Reselect Model & Strategy", type="secondary", width="stretch"):
                    for key in session4init:
                        session_state[key] = None
                    for key in session4pick:
                        session_state[key] = None
                    for key in session4pred:
                        session_state[key] = None
                    rerun()
            else:
                empty_messages.warning(
                    f"You selected a data for prediction. {session_state['timer4pick']} You can repick if needed."
                )

                with original:
                    markdown(f"**The data you selected** - {session_state["idx"]}")
                    write(session_state["label"].squeeze().item())

                if session_state["prediction"] is None:
                    if button("Predict", type="primary", width="stretch"):
                        with Timer("Predict") as session_state["timer4pred"]:
                            session_state["model"].eval()
                            with no_grad():
                                logits: Tensor = session_state["model"](**session_state["sample"])
                                session_state["prediction"]: Tensor = logits.argmax(-1).unsqueeze(1).cpu()
                                rerun()
                else:
                    empty_messages.success(
                        f"Prediction Completed! {session_state["timer4pred"]} You can repredict or repick."
                    )

                    with prediction:
                        markdown(f"**The Prediction Result** - {session_state["idx"]}")
                        write(session_state["prediction"].squeeze().item())

                    display4result.markdown(
                        f"{session_state["idx"]} Sample Prediction Result: {"Bingo" if session_state["label"] == session_state["prediction"] else "Wrong"}"
                    )

                if button("Repick Data", type="secondary", width="stretch"):
                    for key in session4pick:
                        session_state[key] = None
                    for key in session4pred:
                        session_state[key] = None
                    rerun()
    else:
        empty_messages.error("The model & dictionary file does NOT exist.")
