from typing import Any, Optional, cast
import torch

import datasets
import transformers
import numpy
import os

from pipeline import pipeline_protocol


class CutDownTokenizer:
    def __init__(self, base_tokenizer: transformers.PreTrainedTokenizerFast, inverted_id_map: Optional[torch.Tensor] = None):
        self.base_tokenizer = base_tokenizer
        self.base_vocab_size = base_tokenizer.vocab_size

        if inverted_id_map is not None:
            self.inverted_id_map = inverted_id_map
            self.reduced_vocab_size = len(inverted_id_map)
            self.id_map = torch.full((self.base_vocab_size,), fill_value = -1, dtype = torch.long)
            self.id_map[inverted_id_map] = torch.arange(self.reduced_vocab_size, dtype = torch.long)


    def create_id_map(self, split_id_arrays: list[numpy.ndarray]) -> None:
        # use numpy, as it prevents data copying with the pyarrow table representation for huggingface datasets
        merged_unique_ids = numpy.array([], dtype = numpy.int64)

        for id_array in split_id_arrays:
            unique_ids = numpy.unique(id_array, sorted = False)

            merged_unique_ids = numpy.append(merged_unique_ids, unique_ids)

        unique_ids = numpy.unique(merged_unique_ids, sorted = False)

        # convert to torch
        self.inverted_id_map = torch.from_numpy(unique_ids)

        self.reduced_vocab_size = len(self.inverted_id_map)

        self.id_map = torch.full((self.base_vocab_size,), fill_value = -1, dtype = torch.long)
        # remap the unique ids to the first integers
        self.id_map[self.inverted_id_map] = torch.arange(self.reduced_vocab_size, dtype = torch.long)



    def reduce_ids(self, base_ids: torch.Tensor) -> torch.Tensor:
        return self.id_map[base_ids]

    def restore_ids(self, reduced_ids: torch.Tensor) -> torch.Tensor:
        return self.inverted_id_map[reduced_ids]

    def encode(self, text: str) -> torch.Tensor:
        base_ids = self.base_tokenizer.encode(text)
        return self.reduce_ids(torch.tensor(base_ids, dtype = torch.long))

    def decode(self, ids: torch.Tensor) -> str:
        base_ids = self.restore_ids(ids)
        return self.base_tokenizer.decode(base_ids.tolist(), skip_special_tokens = True)




class main_pipeline(pipeline_protocol[datasets.DatasetDict, CutDownTokenizer]):
    def get_dataset_and_tokenizer(self, sequence_length: int, dataset: Optional[datasets.DatasetDict] = None, tokenizer: Optional[CutDownTokenizer] = None) -> tuple[datasets.DatasetDict, CutDownTokenizer]:
        if tokenizer is None:
            base_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            base_tokenizer.pad_token = base_tokenizer.eos_token
        else:
            base_tokenizer = tokenizer.base_tokenizer


        def repack_ids(examples):
            concatenated_ids = sum(examples["input_ids"], [])
            total_length = len(concatenated_ids)

            n_chunks = total_length // sequence_length

            # we need one more token than the number of tokens in the chunks
            # so decrement n_chunks if the single extra token is not available
            if sequence_length * n_chunks + 1 > total_length:
                n_chunks -= 1

            # the last token of the ith chunk is the first token of the (i + 1)th chunk
            return {
                "input_ids": [concatenated_ids[i : i + sequence_length + 1] for i in range(0, n_chunks * sequence_length, sequence_length)]
            }


        if dataset is None:
            dataset = datasets.load_dataset("roneneldan/TinyStories")

            dataset = dataset.map(
                lambda examples: base_tokenizer(examples["text"], truncation = False),
                batched = True,
                remove_columns = ["text"],
                num_proc = 8,
            )

            dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col != "input_ids"])

            dataset = dataset.map(
                repack_ids,
                batched = True,
                num_proc = 8,
            )

            dataset.rename_column("input_ids", "ids")

        if tokenizer is None:
            tokenizer = CutDownTokenizer(base_tokenizer)
            tokenizer.create_id_map([cast(numpy.ndarray, dataset.with_format(type = "numpy")[split]["input_ids"]) for split in dataset.keys()])

        return dataset.with_format(type = "torch"), tokenizer

    def get_dataloaders(self, dataset: datasets.DatasetDict, **dataloader_args: Any) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset["train"]), **dataloader_args)
        val_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset["validation"]), **dataloader_args)

        return train_dataloader, val_dataloader

    def get_training_pairs(self, batch: torch.Tensor, tokenizer: CutDownTokenizer, mask_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = tokenizer.reduce_ids(batch[:, :-1])
        targets = tokenizer.reduce_ids(batch[:, 1:])

        return inputs, targets

    def save_dataset(self, dataset: datasets.DatasetDict, path: str) -> None:
        dataset.save_to_disk(path)

    def load_dataset(self, path: str) -> datasets.DatasetDict:
        return cast(datasets.DatasetDict, datasets.load_from_disk(path))

    def save_tokenizer(self, tokenizer: CutDownTokenizer, path: str) -> None:
        tokenizer.base_tokenizer.save_pretrained(path)
        torch.save(tokenizer.inverted_id_map, os.path.join(path, "inverted_id_map.pt"))

    def load_tokenizer(self, path: str) -> CutDownTokenizer:
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        inverted_id_map = torch.load(os.path.join(path, "inverted_id_map.pt"))

        return CutDownTokenizer(base_tokenizer, inverted_id_map)

    def encode_text(self, tokenizer: CutDownTokenizer, text: str) -> torch.Tensor:
        return tokenizer.encode(text)

    def decode_ids(self, tokenizer: CutDownTokenizer, ids: torch.Tensor) -> str:
        assert ids.dim() == 1, 'ids must be a 1-dimensional tensor'
        return tokenizer.decode(ids)

    def get_vocab_size(self, tokenizer: CutDownTokenizer) -> int:
        return tokenizer.reduced_vocab_size

    def should_halt_generation(self, tokenizer: CutDownTokenizer, last_token_id: int) -> bool:
        return tokenizer.inverted_id_map[last_token_id].item() == tokenizer.base_tokenizer.eos_token_id