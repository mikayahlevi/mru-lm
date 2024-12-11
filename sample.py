import torch

from model import transformer_cache

@torch.no_grad()
# Sample from a starting letter
def sample(model, tokenizer, sequence_start: str, temperature: float, max_length: int, device) -> str:
    
    sequence = [
        torch.tensor(tokenizer.token_to_id(sequence_start), device = device)
    ]
    
    cache = transformer_cache(model.config, device = device)


    for i in range(max_length):
        logits = model.forward(sequence[-1].view(1), cache)
        
        outputs = torch.softmax(logits[-1].squeeze() / temperature, dim=0)

        idx = torch.multinomial(outputs, 1)

        sequence.append(idx)
    
    return tokenizer.decode([i.item() for i in sequence])