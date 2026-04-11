import json
import argparse
from trainer.BaseTrainer import train


def main():
    args = setup_parser().parse_args()
    base_param = load_json(args.base_configs)
    model_param = load_json(args.model_configs)
    merged = {**base_param, **model_param}
    if args.data_root is not None:
        merged['data_root'] = args.data_root
    if args.results_dir is not None:
        merged['results_dir'] = args.results_dir
    train(merged)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Choose your configs.')
    parser.add_argument('--base_configs', type=str, default='./configs/Base_configs/',
                        help='Json file of base settings.')
    parser.add_argument('--model_configs', type=str, default='./configs/model_configs/',
                        help='Json file of model settings.')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data_root from base config.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Override results output directory (default: ~/qz-compcont-learning/results).')

    return parser


if __name__ == '__main__':
    main()
