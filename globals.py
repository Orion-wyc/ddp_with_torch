import argparse
import json


parser = argparse.ArgumentParser(description="argument for DDP training")

parser.add_argument("--conf_path", required=True, type=str,
                    help="prefix identifying training config")


args_global = parser.parse_args()
print(args_global)

if args_global.conf_path is not None:
    with open(args_global.conf_path) as conf_f:
        train_config = json.load(conf_f)
    print(train_config)
