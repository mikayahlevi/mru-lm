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

    if os.path.exists('data/shakespeare_char/dataset') and os.path.exists('data/shakespeare_char/tokenizer'):
        dataset = datasets.load_from_disk('data/shakespeare_char/dataset')
        tokenizer = pickle.load(open('data/shakespeare_char/tokenizer', 'rb'))
    else:
        text = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text
        
        vocab = sorted(set(text))

        tokenizer = character_tokenizer(vocab)

        ids = tokenizer.encode(text)

        train_to_val_ratio = 0.9
        train_len = int(len(ids) * train_to_val_ratio)

        train_ids = ids[:train_len]
        val_ids = ids[train_len:]


        dataset = datasets.DatasetDict({
            'train': datasets.Dataset.from_dict({
                'inputs': train_ids[:-1],
                'labels': train_ids[1:]
            }),
            'validation': datasets.Dataset.from_dict({
                'inputs': val_ids[:-1],
                'labels': val_ids[1:]
            })
        })

        # batch the dataset
        for split in dataset.keys():
            dataset[split] = dataset[split].batch(sequence_length, drop_last_batch = True)

        # save formatted dataset to the disk
        dataset.save_to_disk('data/shakespeare_char/dataset')

        # save tokenizer to the disk
        pickle.dump(tokenizer, open('data/shakespeare_char/tokenizer', 'wb'))

    return dataset.with_format(type = 'torch', columns = ['inputs', 'labels']), tokenizer
        





