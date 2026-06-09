import torch
import argparse
import importlib
import importlib.util
from typing import Any, Protocol, TypeVar, Optional



tokenizer_type = TypeVar('tokenizer_type')
dataset_type = TypeVar('dataset_type')

class pipeline_protocol(Protocol[dataset_type, tokenizer_type]):
    # pass the dataset or tokenizer in to load them
    def get_dataset_and_tokenizer(self, sequence_length: int, dataset: Optional[dataset_type], tokenizer: Optional[tokenizer_type]) -> tuple[dataset_type, tokenizer_type]:
        ...

    def get_dataloaders(self, dataset: dataset_type, **dataloader_args: Any) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        ...

    # if a example has tokens that shouldn't contribute to the loss, they should be replaced with the mask value.
    def get_training_pairs(self, batch: torch.Tensor, tokenizer: tokenizer_type, mask_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def save_dataset(self, dataset: dataset_type, path: str) -> None:
        ...

    def load_dataset(self, path: str) -> dataset_type:
        ...

    def save_tokenizer(self, tokenizer: tokenizer_type, path: str) -> None:
        ...

    def load_tokenizer(self, path: str) -> tokenizer_type:
        ...

    def encode_text(self, tokenizer: tokenizer_type, text: str) -> torch.Tensor:
        ...

    def decode_ids(self, tokenizer: tokenizer_type, ids: torch.Tensor) -> str:
        ...

    def get_vocab_size(self, tokenizer: tokenizer_type) -> int:
        ...

    def should_halt_generation(self, tokenizer: tokenizer_type, last_token_id: int) -> bool:
        ...


def get_pipeline(args: argparse.Namespace) -> pipeline_protocol[Any, Any]:
    if (args.pipeline_name is not None) and (args.pipeline_path is not None):
        raise ValueError('arguments --pipeline_name and --pipeline_path cannot both be set')
    elif args.pipeline_name is not None:
        pipeline_module = importlib.import_module(f'pipelines.{args.pipeline_name}')
    elif args.pipeline_path is not None:
        spec = importlib.util.spec_from_file_location('pipeline_module', args.pipeline_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not load pipeline module from path {args.pipeline_path}.')
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
    else:
        raise ValueError('pipeline must be specified with either --pipeline_name or --pipeline_path argument')

    return pipeline_module.main_pipeline()