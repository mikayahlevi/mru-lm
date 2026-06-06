import torch
import argparse
import importlib
import importlib.util
import colorama

from model import mru_lm_network
from pipeline import pipeline_protocol, get_pipeline
from typing import Any

@torch.inference_mode()
def sample(model, pipeline: pipeline_protocol[Any, Any], tokenizer, prefix: str, temperature: float, max_length: int, device: torch.device) -> str:
    encoded_prefix = pipeline.encode_text(tokenizer, prefix)

    last_state = model.get_initial_state(batch_size = 1, device = device)

    logits = model.forward(encoded_prefix.to(device), last_state).squeeze(-3)

    sequence = torch.empty((max_length,), dtype = torch.long, device = device)

    # generate the first token
    outputs = torch.softmax(logits[-1] / temperature, dim=0)
    last_encoding = torch.multinomial(outputs, 1)
    sequence[0] = last_encoding

    generated = 1

    for _ in range(1, max_length):
        logits = model.forward(last_encoding, last_state).squeeze(-3)

        outputs = torch.softmax(logits[-1] / temperature, dim=0)
        last_encoding = torch.multinomial(outputs, 1)
        sequence[generated] = last_encoding

        generated += 1

        if pipeline.should_halt_generation(tokenizer, int(last_encoding.item())):
            break

    return pipeline.decode_ids(tokenizer, sequence.cpu()[:generated])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from a trained model.')

    parser.add_argument('--model-path', type = str, required = True, help = 'The path to the trained model to sample from.')
    parser.add_argument('--tokenizer-path', type = str, required = True, help = 'The path to the tokenizer used for encoding and decoding text.')
    parser.add_argument('--pipeline-name', type = str, default = None, help = 'The name of the pipeline module used for the tokenizer.')
    parser.add_argument('--pipeline-path', type = str, default = None, help = 'The path to the pipeline module used for the tokenizer.')
    parser.add_argument('--prefix', type = str, default = '', help = 'The prefix to start the generated sequence with.')
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--max-length', type = int, default = 100)
    parser.add_argument('--device', type = str, default = 'cuda')

    args = parser.parse_args()

    pipeline = get_pipeline(args)

    model = torch.load(args.model_path, map_location = args.device, weights_only = False)
    tokenizer = pipeline.load_tokenizer(args.tokenizer_path)

    output = sample(model, pipeline, tokenizer, args.prefix, args.temperature, args.max_length, torch.device(args.device))

    print(colorama.Fore.GREEN)
    print(args.prefix, end = '')
    print(colorama.Fore.YELLOW, end = '')
    print(output)
    print(colorama.Style.RESET_ALL, end = '')