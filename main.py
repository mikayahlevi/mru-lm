import torch

import os
import json
import colorama
import argparse
import prefixed
import contextlib
from typing import Any, Optional, get_origin, Callable

import dataclasses


from model import mru_lm_network, mru_lm_config
from train import train, train_config, hyperparameter_config
from pipeline import pipeline_protocol, get_pipeline


def get_train_folder_path(train_folder_dir: str, train_folder_name: Optional[str]) -> str:
    n_tries = 1000

    if train_folder_name is None:
        train_folder_auto_name_index = 0

        for train_folder_auto_name_index in range(n_tries):
            train_folder_auto_name = 'train-' + str(train_folder_auto_name_index)

            current_auto_path = os.path.join(train_folder_dir, train_folder_auto_name)

            if os.path.exists(current_auto_path):
                train_folder_auto_name_index += 1
            else:
                return current_auto_path

        raise ValueError(f'Could not find a free train folder name in {train_folder_dir} after a maximum of n_tries tries')

    else:
        train_folder_path = os.path.join(train_folder_dir, train_folder_name)
        if os.path.exists(train_folder_path):
            raise ValueError(f'Specified train folder {train_folder_path} already exists')
        return train_folder_path


# load from file, load from arguments, add overrides, verify, and re-save the configs
def get_config(args: argparse.Namespace, cfg_type: type[Any], cfg_name: str, overrides: dict[str, Any], train_folder_path: str) -> Any:
    cfg_path = getattr(args, f'{cfg_name}_cfg_path')

    cfg_dict = {}

    if cfg_path is not None:
        if not os.path.exists(cfg_path):
            raise ValueError(f'{cfg_type} config path {cfg_path} does not exist')

        with open(cfg_path, 'r') as f:
            cfg_dict.update(json.load(f))

    # load the fields from the cli arguments
    for field in dataclasses.fields(cfg_type):
        if hasattr(args, field.name) and getattr(args, field.name) is not None:
            cfg_dict[field.name] = getattr(args, field.name)

    # verify the config dict does not have unexpected fields
    # assume the overrides cannot have unexpected fields
    for key in cfg_dict:
        if key not in [field.name for field in dataclasses.fields(cfg_type)]:
            raise ValueError(f'{cfg_name} config has unexpected field {key} that is not defined in the config dataclass')
        # check that none of the overridden fields are set
        if key in overrides:
            raise ValueError(f'{cfg_name} config field {key} should not be manually set')

    # apply the overrides
    for key, value in overrides.items():
        cfg_dict[key] = value

    # verify the config dict has all the required fields and that they are of the correct type
    for field in dataclasses.fields(cfg_type):
        if field.name not in cfg_dict:
            raise ValueError(f'{cfg_name} config is missing required field {field.name}')

        # skip verifying the type because isinstance does not work with generics, such a list[int]

        # if not isinstance(cfg_dict[field.name], field.type):
        #     raise ValueError(f'{cfg_name} config field {field.name} is of type {type(cfg_dict[field.name])}, but expected type is {field.type}')

    # save the config dict as a json in the train folder
    with open(os.path.join(train_folder_path, f'{cfg_name}.json'), 'w') as f:
        f.write(json.dumps(cfg_dict))

    return cfg_type(**cfg_dict)



def add_typed_arguments(parser: argparse.ArgumentParser, arg_str: str, type: Any):
    origin = get_origin(type)

    if origin == list:
        parser.add_argument(arg_str, type = lambda s: [type.__args__[0](item) for item in s.split(',')])
    if origin == tuple:
        parser.add_argument(arg_str, type = lambda s: tuple(type.__args__[0](item) for item in s.split(',')))
    else:
        parser.add_argument(arg_str, type = type)


def manage_dataset_and_tokenizer(args: argparse.Namespace, pipeline: pipeline_protocol[Any, Any], sequence_length: int) -> tuple[Any, Any]:
    dataset, tokenizer = None, None

    if args.dataset_load_path is not None:
        if not os.path.exists(args.dataset_load_path):
            raise ValueError(f'dataset load path {args.dataset_load_path} does not exist')

        dataset = pipeline.load_dataset(args.dataset_load_path)
        print(colorama.Fore.BLUE)
        print(f'loaded dataset from {args.dataset_load_path}')
        print(colorama.Style.RESET_ALL, end='')
    if args.tokenizer_load_path is not None:
        if not os.path.exists(args.tokenizer_load_path):
            raise ValueError(f'tokenizer load path {args.tokenizer_load_path} does not exist')

        tokenizer = pipeline.load_tokenizer(args.tokenizer_load_path)
        print(colorama.Fore.BLUE)
        print(f'loaded tokenizer from {args.tokenizer_load_path}')
        print(colorama.Style.RESET_ALL, end='')

    if dataset is None or tokenizer is None:
        dataset, tokenizer = pipeline.get_dataset_and_tokenizer(sequence_length = train_cfg.sequence_length, dataset = dataset, tokenizer = tokenizer)

    # only save if they weren't loaded
    if args.dataset_save_path is not None and args.dataset_load_path is None:
        pipeline.save_dataset(dataset, args.dataset_save_path)
    if args.tokenizer_save_path is not None and args.tokenizer_load_path is None:
        pipeline.save_tokenizer(tokenizer, args.tokenizer_save_path)

    return dataset, tokenizer



def manage_wandb(args: argparse.Namespace, train_cfg: train_config, hprms_cfg: hyperparameter_config, model_cfg: transformer_config) -> tuple[contextlib.AbstractContextManager, Callable[[int, str, float], None]]:
    # initialize wandb logging if enabled
    # uses optional context managers and metric logging functions
    wandb_cm = contextlib.nullcontext()
    wandb_log_metric = lambda step, metric, value: None

    if args.log_to_wandb:
        import wandb
        import getpass

        print(colorama.Fore.BLUE)
        print('logging via wandb enabled')
        print(colorama.Style.RESET_ALL, end='')

        wandb.login(key = os.environ.get("WANDB_API_KEY") or getpass.getpass("enter wandb key: "))


        wandb_cm = wandb.init(
            project = os.environ.get("WANDB_PROJECT") or input("enter wandb project name: "),
            name = os.environ.get("WANDB_NAME") or input("enter wandb run name: "),
            config = {
                **dataclasses.asdict(train_cfg),
                **dataclasses.asdict(hprms_cfg),
                **dataclasses.asdict(model_cfg)
            }
        )

        # wandb uses 0-indexing for the step
        wandb_log_metric = lambda step, metric, value: wandb.log({metric: value}, step = step + 1)

    return wandb_cm, wandb_log_metric



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type = str, default = 'cuda')

    parser.add_argument('--precision', type = str, choices = ['fp16', 'bf16', 'fp32'], default = None, help = 'set to fp16 or bf16 to enable mixed precision. otherwise defaults to fp32')


    parser.add_argument('--pipeline-name', type = str, default = None)
    parser.add_argument('--pipeline-path', type = str, default = None)

    parser.add_argument('--compile', action = 'store_true')

    parser.add_argument('--train-folder-dir', type = str, default = 'trains', help = 'directory to create the train folder in')
    parser.add_argument('--train-folder-name', type = str, default = None, help = 'name of folder to save files and logs')

    parser.add_argument('--dataset-save-path', type = str, default = None)
    parser.add_argument('--tokenizer-save-path', type = str, default = None)

    parser.add_argument('--dataset-load-path', type = str, default = None)
    parser.add_argument('--tokenizer-load-path', type = str, default = None)


    parser.add_argument('--loss-log-interval', type = int, default = 20, help = 'steps between logging the training loss')

    parser.add_argument('--eval-interval', type = int, default = 200, help = 'steps between running the model on the validation dataset and logging the resulting loss')

    parser.add_argument('--log-to-file', action = 'store_true')
    parser.add_argument('--log-to-wandb', action = 'store_true')


    parser.add_argument('--metric-print-interval', type = int, default = 40)

    parser.add_argument('--checkpoint-save-interval', type = int, default = 1000)


    parser.add_argument('--model-cfg-path', type = str, default = None)
    parser.add_argument('--train-cfg-path', type = str, default = None)
    parser.add_argument('--hprms-cfg-path', type = str, default = None)

    parser.add_argument('--checkpoint-load-path', type = str, default = None, help = 'path to a checkpoint to load and resume training from')

    # add overrides for all config values as cli arguments
    for config in [train_config, hyperparameter_config, mru_lm_config]:
        for field in dataclasses.fields(config):
            if config == mru_lm_config and field.name == 'vocab_size':
                continue # vocab size is determined by the pipeline and should not be set manually

            # format the names to replace underscores with dashes for the cli arguments
            add_typed_arguments(parser, f'--{field.name.replace("_", "-")}', field.type)


    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(colorama.Fore.YELLOW)
        print('warning: unknown arguments:', unknown_args)
        print(colorama.Style.RESET_ALL, end='')


    # choose and create the path to log the info
    train_folder_path = get_train_folder_path(args.train_folder_dir, args.train_folder_name)
    os.makedirs(train_folder_path)

    checkpoint_save_path = os.path.join(train_folder_path, 'checkpoints')
    os.makedirs(checkpoint_save_path)

    log_path = os.path.join(train_folder_path, 'log.txt')


    # load the model config later so that the vocab size can be overriden
    # vocab size is determined after sequence length is needed from the train config
    train_cfg, hprms_cfg = get_config(args, train_config, 'train', {}, train_folder_path), get_config(args, hyperparameter_config, 'hprms', {}, train_folder_path)

    pipeline = get_pipeline(args)

    dataset, tokenizer = manage_dataset_and_tokenizer(args, pipeline, sequence_length = train_cfg.sequence_length)


    model_cfg = get_config(args, mru_lm_config, 'model', {'vocab_size': pipeline.get_vocab_size(tokenizer)}, train_folder_path = train_folder_path)


    wandb_cm, wandb_log_metric = manage_wandb(args, train_cfg, hprms_cfg, model_cfg)


    def log_metric(step, metric, value):
        wandb_log_metric(step, metric, value)

        if args.log_to_file:
            with open(log_path, 'a') as file:
                # use 1-indexing for the step in the log file
                file.write(f'step: {step + 1}  {metric}: {value:.6f}\n')


    checkpoint = None
    if args.checkpoint_load_path is not None:
        if not os.path.exists(args.checkpoint_load_path):
            raise ValueError(f'checkpoint load path {args.checkpoint_load_path} does not exist')

        checkpoint = torch.load(args.checkpoint_load_path, map_location = args.device)
        print(colorama.Fore.BLUE)
        print(f'loaded checkpoint from {args.checkpoint_load_path}')
        print(colorama.Style.RESET_ALL, end='')


    model = mru_lm_network(model_cfg).to(args.device)

    # load the state dict before training, everything else is loaded in the train function
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])


    print(colorama.Fore.BLUE)
    print('model size:', f'{prefixed.Float(sum(p.numel() for p in model.parameters())):.2h}', 'parameters')
    print(colorama.Style.RESET_ALL, end='')
    if args.compile:
        model = torch.compile(model)


    with wandb_cm:
        train(
            train_cfg,
            hprms_cfg,

            model,

            checkpoint,

            dataset,

            tokenizer,

            pipeline,

            log_metric,

            args.loss_log_interval,
            args.eval_interval,
            args.metric_print_interval,
            args.checkpoint_save_interval,

            checkpoint_save_path,

            args.device,

            args.precision
        )