import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)

# SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("SRC_DIR:", SRC_DIR)
print("sys.path", sys.path)

parser = argparse.ArgumentParser(description="argument for DDP training")

parser.add_argument("--conf_path", required=True, type=str, help="prefix identifying training config")
parser.add_argument("--eval_internal", default=5, type=int, help="evaluate model every n epochs")
parser.add_argument("--gpu", default=-1, type=str, help="train with GPU(s), use ',' to seperate, default=-1")

args_global = parser.parse_args()
print(args_global)

if args_global.conf_path is not None:
    with open(args_global.conf_path) as conf_f:
        train_config = json.load(conf_f)
    print(train_config)
