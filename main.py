#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/2 22:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from app.tools.layout import config_page, set_pages


def main() -> None:
    """ streamlit run main.py """
    config_page()
    set_pages()


if __name__ == "__main__":
    main()
