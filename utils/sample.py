import torch
import argparse
import colorama
import json

from model import mru_lm_network
from pipeline import pipeline_protocol, get_pipeline
from typing import Any



def sample(model: mru_lm_network, pipeline: pipeline_protocol[Any, Any], tokenizer, prefix: str, temperature: float, max_length: int, device: torch.device, print_output: bool = False) -> str:
    sequence = torch.empty((max_length,), dtype = torch.long, device = device)

    prev_encoding = pipeline.encode_text(tokenizer, prefix).to(device)

    last_state = model.get_initial_state()

    generated = max_length
    for index in range(0, max_length):
        logits = model(prev_encoding, last_state)
        outputs = torch.softmax(logits[..., -1, :] / temperature, dim = -1)

        prev_encoding = torch.multinomial(outputs, 1)

        sequence[index] = prev_encoding

        if print_output:
            print(pipeline.decode_ids(tokenizer, prev_encoding.cpu()))
        if pipeline.should_halt_generation(tokenizer, int(prev_encoding.item())):
            generated = index + 1
            break

    return pipeline.decode_ids(tokenizer, sequence.cpu()[:generated])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample from a trained model')

    parser.add_argument('--model-path', type = str, required = True, help = 'the path to the trained model to sample from')
    parser.add_argument('--checkpoint-path', type = str, default = None, help = 'the path to the checkpoint to load')

    parser.add_argument('--config-path', type = str, default = None, help = 'the path to the model configuration file')

    parser.add_argument('--tokenizer-path', type = str, required = True, help = 'the path to the tokenizer used for encoding and decoding text')

    parser.add_argument('--pipeline-name', type = str, default = None, help = 'the name of the pipeline module used for the tokenizer')
    parser.add_argument('--pipeline-path', type = str, default = None, help = 'the path to the pipeline module used for the tokenizer')

    parser.add_argument('--prefix', type = str, default = '', help = 'the prefix to start the generated sequence with')
    parser.add_argument('--stream', type = bool, default = True, help = 'Wwether to print the output as it is generated')
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--max-length', type = int, default = 100)

    parser.add_argument('--device', type = str, default = 'cuda')

    args = parser.parse_args()

    pipeline = get_pipeline(args)

    if args.model_path is None and args.checkpoint_path is None:
        raise ValueError("either --model-path or --checkpoint-path must be specified")
    elif args.model_path is not None:
        model = torch.load(args.model_path, map_location = args.device, weights_only = False)
    elif args.checkpoint_path is not None:
        if args.config_path is None:
            raise ValueError("when using --checkpoint-path, --config-path must also be specified")

        with open(args.config_path, 'r') as f:
            config = json.load(f)

        checkpoint = torch.load(args.checkpoint_path, map_location = args.device, weights_only = False)
        model_state_dict = checkpoint['model_state_dict']
        model = mru_lm_network(config)
        model.load_state_dict(model_state_dict)


    tokenizer = pipeline.load_tokenizer(args.tokenizer_path)


    print(colorama.Fore.GREEN)
    print(args.prefix, end = '')

    print(colorama.Fore.YELLOW, end = '')


    output = sample(model, pipeline, tokenizer, args.prefix, args.temperature, args.max_length, torch.device(args.device), args.stream)

    if not args.stream:
        print(output)

    print(colorama.Style.RESET_ALL, end = '')