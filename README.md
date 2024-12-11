# transformer-train-script

## What is this?
A little PyTorch script I made to train a transformer.
Currently supports the tiny_stories dataset, and all of shakespeare as a text file, using each character as a token.

## How to Use
Use the command `python main.py --device=cuda --dataset=tiny_stories.py`. Set `--dataset=shakespeare_char` for the character-level shakespeare dataset.
