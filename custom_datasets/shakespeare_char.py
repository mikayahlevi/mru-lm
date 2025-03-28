import requests
import datasets
import pickle

import os


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


def get_dataset_and_tokenizer(sequence_length: int):
    if not os.path.exists('data/shakespeare_char'):
        os.makedirs('data/shakespeare_char')

    dataset, vocab = None, None

    text = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text
    
    vocab = sorted(set(text))

    tokenizer = character_tokenizer(vocab)

    ids = tokenizer.encode(text)

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
        





