import torch

import os
import json
import colorama
import argparse
import prefixed
import importlib
from typing import Optional

import dataclasses


from model import mru_lm_network, mru_lm_config
from train import train, train_config, hyperparameter_config



parser = argparse.ArgumentParser()

parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--dataset', type = str)
parser.add_argument('--compile', type = bool, default = False)
parser.add_argument('--config_path', type = str, default = 'config')
parser.add_argument('--train_log_path', type = str, default ='trains')


args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')


    # load the dataset module
    dataset_module = importlib.import_module(f'custom_datasets.{args.dataset}')


    # load the configs from the json files
    traincfg, modelcfg, hparams = None, None, None
    with open(os.path.join(args.config_path, 'traincfg.json'), 'r') as f:
        traincfg = train_config(**json.load(f))
    with open(os.path.join(args.config_path, 'hparams.json'), 'r') as f:
        hparams = hyperparameter_config(**json.load(f))



    dataset, tokenizer = dataset_module.get_dataset_and_tokenizer(traincfg.sequence_length)


    with open(os.path.join(args.config_path, 'modelcfg.json'), 'r') as f:
        # set the model's vocab size to the dataset's vocab size
        modelcfg = mru_lm_config(**json.load(f), vocab_size = tokenizer.get_vocab_size())


    # create the path to log the info and dump the configs as jsons
    train_folder_index = 0
    while True:
        current_path = os.path.join(args.train_log_path, 'train-' + str(train_folder_index))
        if os.path.exists(current_path):
            train_folder_index += 1
        else:
            os.makedirs(current_path)
            traincfg.train_log_path = current_path

            os.makedirs(traincfg.train_log_path  + '/models')
            os.makedirs(traincfg.train_log_path  + '/stats')

            with open(traincfg.train_log_path  + '/traincfg.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(traincfg)))
            with open(traincfg.train_log_path  + '/hparams.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(hparams)))
            with open(traincfg.train_log_path  + '/modelcfg.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(modelcfg)))

            break






    model = (torch.compile(mru_lm_network(modelcfg)) if args.compile else mru_lm_network(modelcfg)).to(args.device)


    print(colorama.Fore.BLUE)
    print('parameters:', f'{prefixed.Float(sum(p.numel() for p in model.parameters())):.2h}', 'parameters')
    print(colorama.Style.RESET_ALL, end='')



    train(
        traincfg,
        hparams,
        model,
        dataset,
        args.device
    )