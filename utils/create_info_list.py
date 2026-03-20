#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import re
import argparse
import json

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


def get_celeb_imdb(args):
    celeb_names = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            result = re.search(r"^\d+\.\s*(.+)", line.strip())

            if result:
                celeb_names.append(result.group(1))
    print(len(celeb_names))

    with open("./result.txt", "w", encoding="utf-8") as f:
        for line in celeb_names:
            f.write(line + "\n")


def get_celeb_chatgpt(args, key="singer"):
    celeb_names = {key: []}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                # result = re.search(r"^(.*?)\s*\(", line)
                result = re.search(r"^(.+)\s*\(", line)

                if result:
                    celeb_names[key].append(result.group(1).strip())
                else:
                    celeb_names[key].append(line)

    print(len(celeb_names[key]))

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(celeb_names, f, indent=4, ensure_ascii=False)



def get_makeup_chatgpt(args):
    makeup_info = {}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                result = re.match(r"^(.*?):", line)
                result2 = re.match(r"^.*?:\s*(.*)", line)

                if result:
                    makeup_info[key][result.group(1).strip()] = result2.group(1).strip()
                else:
                    key = line
                    makeup_info[key] = {}

    print(len(makeup_info[key]))

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(makeup_info, f, indent=4, ensure_ascii=False)


def get_racename_chatgpt(args, key="Asian"):
    name_dict = {key: []}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                result = re.search(r"^\d+\.\s*(.+)", line)

                if result:
                    name_dict[key].append(result.group(1).strip())
                else:
                    name_dict[key].append(line)

    print(len(name_dict[key]))

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(name_dict, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument("--data", type=str, default="./data.txt", help="path to dataset")
    args = parser.parse_args()

    get_makeup_chatgpt(args)

    # get_racename_chatgpt(args)