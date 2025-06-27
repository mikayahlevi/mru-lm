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
parser.add_argument('--config_folder_path', type = str, default = 'config')
parser.add_argument('--train_folder_dir', type = str, default = 'trains')
parser.add_argument('--train_folder_name', type = Optional[str], default = None)


args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')


    # load the dataset module
    dataset_module = importlib.import_module(f'custom_datasets.{args.dataset}')


    # load the configs from the json files
    traincfg, modelcfg, hparams = None, None, None
    with open(os.path.join(args.config_folder_path, 'traincfg.json'), 'r') as f:
        traincfg = train_config(**json.load(f))
    with open(os.path.join(args.config_folder_path, 'hparams.json'), 'r') as f:
        hparams = hyperparameter_config(**json.load(f))


    dataset, tokenizer = dataset_module.get_dataset_and_tokenizer(traincfg.sequence_length)

    with open(os.path.join(args.config_folder_path, 'modelcfg.json'), 'r') as f:
        # set the model's vocab size to the dataset's vocab size
        modelcfg = mru_lm_config(**json.load(f), vocab_size = tokenizer.get_vocab_size())


    # create the path to log the info and dump the configs as jsons
    train_folder_path = None

    if args.train_folder_name is None:
        train_folder_auto_name_index = 0
        
        for train_folder_auto_name_index in range(1000):
            train_folder_auto_name = 'train-' + str(train_folder_auto_name_index)
            
            current_auto_path = os.path.join(args.train_folder_dir, train_folder_auto_name)
            
            if os.path.exists(current_auto_path):
                train_folder_auto_name_index += 1
            else:
                train_folder_path = current_auto_path
                break
        
        if train_folder_path is None:
            raise ValueError(f'Could not find a free train folder name in {args.train_folder_dir} after a maximum of 1000 tries.')

    else:
        train_folder_path = os.path.join(args.train_folder_dir, args.train_folder_name)
        if os.path.exists(train_folder_path):
            raise ValueError(f'Specified train folder {train_folder_path} already exists.')

    os.makedirs(train_folder_path)

    os.makedirs(os.path.join(train_folder_path, 'models'))
    os.makedirs(os.path.join(train_folder_path, 'stats'))

    with open(os.path.join(train_folder_path, 'traincfg.json'), 'w') as f:
        f.write(json.dumps(dataclasses.asdict(traincfg)))
    with open(os.path.join(train_folder_path, 'hparams.json'), 'w') as f:
        f.write(json.dumps(dataclasses.asdict(hparams)))
    with open(os.path.join(train_folder_path, 'modelcfg.json'), 'w') as f:
        f.write(json.dumps(dataclasses.asdict(modelcfg)))
    
    
    traincfg.train_folder_path = train_folder_path





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