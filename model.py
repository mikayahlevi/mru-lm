import torch

import math

from typing import Optional
from dataclasses import dataclass

    


@dataclass
class mrun_block_config:
    n_state_heads: int
    state_size: int

    key_size: int
    value_size: int

    n_mlp_layers: int



@dataclass
class mrun_network_config:
    vocab_size: int
    embedding_size: int

    dropout_rate: float

    max_sequence_length: int
    
    block_configs: list[mrun_block_config]


class flat_relu_mlp(torch.nn.Module):
    def __init__(self, intermediate_size, n_intermediate_layers):
        super(flat_relu_mlp, self).__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(intermediate_size, intermediate_size, bias = False) for _ in range(n_intermediate_layers)
        ])

        expected_mean = 0.398942280401
        expected_std = 0.583796851386

        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                torch.nn.init.normal_(layer.weight, mean = 0, std = 1 / math.sqrt(intermediate_size))
            else:
                torch.nn.init.normal_(layer.weight, mean = 0, std = 1 / math.sqrt((expected_mean ** 2 + expected_std ** 2) * intermediate_size))

        self.mean_offset_constant = torch.nn.Parameter(torch.tensor([-expected_mean]), requires_grad=False)
        self.std_scale_constant = torch.nn.Parameter(torch.tensor([1 / expected_std]), requires_grad=False)
    
    def forward(self, input):
        for layer in self.layers:
            input = torch.nn.functional.relu(layer(input))

        return (input + self.mean_offset_constant) * self.std_scale_constant



class mrun_block(torch.nn.Module):
    def __init__(self, network_config: mrun_network_config, block_config: mrun_block_config):
        super(mrun_block, self).__init__()

        self.block_config = block_config
        self.network_config = network_config


        if block_config.state_size % block_config.n_state_heads != 0:
            raise ValueError("state size must be divisible by the number of state heads")
        self.state_size = block_config.hidden_size // block_config.n_attn_heads



        hidden_scale = 1 / math.sqrt(network_config.embedding_size)
        attention_scale = 0.25
        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = attention_scale * hidden_scale)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = attention_scale * hidden_scale)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = hidden_scale)


        self.attention_down = torch.nn.Linear(block_config.value_size, network_config.embedding_size, bias = False)
        torch.nn.init.normal_(self.attention_down.weight, mean = 0, std = 1 / math.sqrt(block_config.value_size))
        

        self.mlp = flat_relu_mlp(network_config.embedding_size, block_config.n_mlp_layers)


        self.residule_scale = torch.nn.Parameter(torch.tensor([1 / math.sqrt(2)]), requires_grad=False)

        self.initial_state = torch.nn.Parameter(torch.normal(mean = 0, std = 1, size = (block_config.n_state_heads, block_config.state_size)), requires_grad=True)
    
    def parallel_mru():
        pass

    def forward(self, activations: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = torch.nn.functional.dropout(activations, p = self.network_config.dropout_rate, training = self.training)
        # kv_cache is modified in place
        activations = norm(activations + self.parallel_mru())
        activations = norm(activations + self.mlp(activations))
        return activations, state




    
class mrun_network(torch.nn.Module):
    def __init__(self, config: mrun_network_config):
        super(mrun_network, self).__init__()

        self.config = config

        
        self.blocks = torch.nn.ModuleList([mrun_block(config, block_config) for block_config in config.block_configs])
        
        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 1)
        
        self.lm_head_weights = self.wte.weight

        self.embedding_scale_constant = torch.nn.Parameter(torch.tensor([1 / math.sqrt(config.embedding_size)]), requires_grad=False)

    # index should start at 0
    def forward(self, encodings: torch.Tensor, state: list[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:

        embeddings = self.wte(encodings)

        for i, block in enumerate(self.blocks):
            embeddings, state[i] = block.forward(embeddings, state[i])
        
        logits = torch.nn.functional.linear(embeddings, weight = self.lm_head_weights * self.embedding_scale_constant)

        return logits, state
        
    
    def get_initial_state(self) -> list[torch.Tensor]:
        return ([block.initial_state for block in self.blocks])
