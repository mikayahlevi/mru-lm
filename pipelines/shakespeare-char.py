import requests
import datasets
import torch
import pickle
from typing import cast, Any, Optional


from pipeline import pipeline_protocol

class character_tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.token_to_id_dict = {token: i for i, token in enumerate(vocab)}
        self.id_to_token_dict = {i: token for i, token in enumerate(vocab)}

        self.padding = False

        self.character_level = True

    def encode(self, text):
        return [self.token_to_id_dict[token] for token in text]

    def token_to_id(self, token):
        return self.token_to_id_dict[token]

    def decode(self, ids):
        return ''.join([self.id_to_token_dict[i] for i in ids])

    def get_vocab_size(self):
        return self.vocab_size


class main_pipeline(pipeline_protocol[datasets.DatasetDict, character_tokenizer]):
    def get_dataset_and_tokenizer(self, sequence_length: int, dataset: Optional[datasets.DatasetDict] = None, tokenizer: Optional[character_tokenizer] = None) -> tuple[datasets.DatasetDict, character_tokenizer]:
        if tokenizer is None or dataset is None:
            text = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text

        if tokenizer is None:
            vocab = sorted(set(text)) # type: ignore

            tokenizer = character_tokenizer(vocab)

        if dataset is None:
            ids = tokenizer.encode(text) # type: ignore

            train_to_val_ratio = 0.9
            train_len = int(len(ids) * train_to_val_ratio)


            dataset = datasets.DatasetDict({
                'train': datasets.Dataset.from_dict({
                    'ids': ids[:train_len]
                }),
                'validation': datasets.Dataset.from_dict({
                    'ids': ids[train_len:]
                })
            })

            # batch the dataset
            for split in dataset.keys():
                dataset[split] = dataset[split].batch(sequence_length + 1, drop_last_batch = True)

        return dataset.with_format(type = 'torch', columns = ['ids']), tokenizer

    def get_dataloaders(self, dataset: datasets.DatasetDict, **dataloader_args: Any) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset['train']), **dataloader_args)
        val_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset['validation']), **dataloader_args)

        return train_dataloader, val_dataloader

    def get_training_pairs(self, batch: torch.Tensor, tokenizer: character_tokenizer, mask_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        return inputs, targets

    def save_dataset(self, dataset: datasets.DatasetDict, path: str):
        dataset.save_to_disk(path)

    def load_dataset(self, path: str) -> datasets.DatasetDict:
        return datasets.DatasetDict.load_from_disk(path)

    def save_tokenizer(self, tokenizer: character_tokenizer, path: str):
        # dump the vocab
        with open(path, 'wb') as f:
            pickle.dump(tokenizer.vocab, f)

    def load_tokenizer(self, path: str) -> character_tokenizer:
        with open(path, 'rb') as f:
            vocab = pickle.load(f)

        return character_tokenizer(vocab)

    def encode_text(self, tokenizer: character_tokenizer, text: str) -> torch.Tensor:
        return torch.Tensor(tokenizer.encode(text)).long()

    def decode_ids(self, tokenizer: character_tokenizer, ids: torch.Tensor) -> str:
        assert ids.dim() == 1, 'ids must be a 1-dimensional tensor'
        return tokenizer.decode(ids.tolist())

    def get_vocab_size(self, tokenizer: character_tokenizer) -> int:
        return tokenizer.get_vocab_size()

    def should_halt_generation(self, tokenizer: character_tokenizer, last_token_id: int) -> bool:
        return False