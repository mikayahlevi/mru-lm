import datasets
import tokenizers
import os


def get_dataset_and_tokenizer(sequence_length: int):
    if not os.path.exists('data/tiny_stories'):
        os.makedirs('data/tiny_stories')

    if os.path.exists('data/tiny_stories/dataset') and os.path.exists('data/tiny_stories/tokenizer'):
        dataset = datasets.load_from_disk('data/tiny_stories/dataset')
        tokenizer = tokenizers.Tokenizer.from_file('data/tiny_stories/tokenizer')

    else:
        
        dataset = datasets.load_dataset('roneneldan/TinyStories')
        
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_token = '<unk>'))
        tokenizer.normalizer = tokenizers.normalizers.NFKC()
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        # vocab size is currently hardcoded
        trainer = tokenizers.trainers.WordPieceTrainer(vocab_size = 8192, special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>'])


        # train the tokenizer on the train split
        tokenizer.train_from_iterator(dataset['train']['text'], trainer = trainer)

        tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single = '<bos> $A <eos>',
            special_tokens = [
                ('<bos>', tokenizer.token_to_id('<bos>')),
                ('<eos>', tokenizer.token_to_id('<eos>'))
            ]
        )
            
        tokenizer.enable_padding(pad_id = tokenizer.token_to_id('<pad>'), pad_token = '<pad>', length = sequence_length + 1)
        tokenizer.enable_truncation(max_length = sequence_length  + 1)



        # print(tokenizer.encode('Let\'s test this tokenizer.').tokens)

        # tokenize the dataset
        for split in dataset.keys():

            def process(example):
                tokens = tokenizer.encode(example['text']).ids
                return {'inputs': tokens[:-1], 'labels': tokens[1:]}

            # tokenize the text
            # batched = True doesn't work for some reason
            dataset[split] = dataset[split].map(process, remove_columns = ['text'])

            
        # save the dataset and tokenizer
        tokenizer.save('data/tiny_stories/tokenizer')
        dataset.save_to_disk('data/tiny_stories/dataset')


    return dataset.with_format(type = 'torch', columns = ['inputs', 'labels']), tokenizer