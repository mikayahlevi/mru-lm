import torch

import math

from typing import Optional
from dataclasses import dataclass

    


@dataclass
class transformer_block_config:
    n_attn_heads: int

    hidden_size: int
    key_size: int
    value_size: int



@dataclass
class transformer_network_config:
    vocab_size: int
    embedding_size: int

    dropout_rate: float

    max_sequence_length: int
    
    block_configs: list[transformer_block_config]



class xpos(torch.nn.Module):
    def __init__(self, key_head_size: int, max_sequence_length: int = 1024):
        super(xpos, self).__init__()

        if key_head_size % 2 != 0:
            raise ValueError("key head size must be divisible by 2 for the positional embedding")

        theta_base = 10000
        alpha = 0.4 * key_head_size

        drange = torch.arange(start = 2, end = key_head_size + 2, step = 2, dtype = torch.float32)
        theta = torch.pow(1 / theta_base, drange / key_head_size).repeat_interleave(2)
        zeta = ((drange / (key_head_size / 2) + alpha) / (1 + alpha)).repeat_interleave(2)
        # no effect except for numerical stability
        scale_base = 512
        # no effect except for numerical stability
        half_max_sequence_length = max_sequence_length // 2

        seq_range = torch.arange(- half_max_sequence_length, max_sequence_length - half_max_sequence_length, dtype = torch.float32).view(-1, 1, 1) / scale_base

        self.c = torch.nn.Parameter(torch.cos(seq_range * theta.view(1, 1, -1)), requires_grad=False)
        self.s = torch.nn.Parameter(torch.sin(seq_range * theta.view(1, 1, -1)), requires_grad=False)
        self.t = torch.nn.Parameter((zeta.view(1, 1, -1) ** seq_range), requires_grad=False)
        self.invt = torch.nn.Parameter(1 / self.t, requires_grad=False)



    def rotate_every_two(self, input: torch.Tensor) -> torch.Tensor:
        return torch.stack((-input[..., 1::2], input[..., 0::2]), dim = -1).flatten(-2)


    def forward(self, queries, keys, start, end) -> tuple[torch.Tensor, torch.Tensor]:
        queries = (queries * self.c[start:end] + self.rotate_every_two(queries) * self.s[start:end]) * self.t[start:end]
        keys = (keys * self.c[start:end] + self.rotate_every_two(keys) * self.s[start:end]) * self.invt[start:end]


        return queries, keys


class flat_elu_mlp(torch.nn.Module):

    def __init__(self, intermediate_size, n_intermediate_layers):
        super(flat_elu_mlp, self).__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(intermediate_size, intermediate_size, bias = False) for _ in range(n_intermediate_layers)
        ])

        for layer in self.layers:
            torch.nn.init.normal_(layer.weight, mean = 0, std = 1 / math.sqrt(intermediate_size))

        self.mean_constant = torch.nn.Parameter(torch.tensor([0.160520572266]), requires_grad=False)
        self.std_constant = torch.nn.Parameter(torch.tensor([0.786879001735]), requires_grad=False)
    
    def forward(self, input):
        for layer in self.layers:
            input = (torch.nn.functional.elu(layer(input)) - self.mean_constant) / self.std_constant
        return input



class transformer_block(torch.nn.Module):
    def __init__(self, network_config: transformer_network_config, block_config: transformer_block_config):
        super(transformer_block, self).__init__()

        self.block_config = block_config
        self.network_config = network_config



        if block_config.key_size % block_config.n_attn_heads != 0:
            raise ValueError("key size must be divisible by the number of attention heads")
        self.key_head_size = block_config.hidden_size // block_config.n_attn_heads

        if block_config.value_size % block_config.n_attn_heads != 0:
            raise ValueError("value size must be divisible by the number of attention heads")
        self.value_head_size = block_config.hidden_size // block_config.n_attn_heads



        self.first_ln = torch.nn.LayerNorm(block_config.hidden_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(block_config.hidden_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(block_config.hidden_size, block_config.hidden_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(block_config.hidden_size * 4, block_config.hidden_size, bias = False)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))


        self.query_layer = torch.nn.Linear(block_config.hidden_size, block_config.key_size, bias = False)
        self.key_layer = torch.nn.Linear(block_config.hidden_size, block_config.key_size, bias = False)
        self.value_layer = torch.nn.Linear(block_config.hidden_size, block_config.value_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 0.02)

        self.attention_linear = torch.nn.Linear(block_config.value_size, block_config.hidden_size, bias = False)
        torch.nn.init.normal_(self.attention_linear.weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))

        self.position_embedding = xpos(self.key_head_size, max_sequence_length = network_config.max_sequence_length)

        self.sqrt_two_constant = torch.nn.Parameter(torch.tensor([math.sqrt(2)]), requires_grad=False)
    

    def get_full_kv(self, incoming_kv, kv_cache, index) -> tuple[tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if kv_cache is None:
            return incoming_kv, None
        else:
            incoming_keys, incoming_values = incoming_kv
            cache_keys, cache_values = kv_cache

            incoming_sequence_length = incoming_keys.size(-3)
            cache_sequence_length = index
            total_sequence_length = incoming_sequence_length + cache_sequence_length
            max_sequence_length = cache_keys.size(-3)

            if total_sequence_length > max_sequence_length:
                raise ValueError("total sequence length is larger than the maximum sequence length of the cache")
            elif total_sequence_length == max_sequence_length and index == 0:
                return (incoming_keys, incoming_values), None
            elif total_sequence_length == max_sequence_length:
                keys = torch.cat((cache_keys[..., :cache_sequence_length, :, :], incoming_keys), dim = -3)
                values = torch.cat((cache_values[..., :cache_sequence_length, :, :], incoming_values), dim = -3)

                mask = torch.ones(incoming_sequence_length, total_sequence_length, dtype = torch.bool, device = keys.device).triu()

                return (keys, values), mask
            elif index == 0:
                cache_keys[..., :incoming_sequence_length, :, :] = incoming_keys
                cache_values[..., :incoming_sequence_length, :, :] = incoming_values

                return (incoming_keys, incoming_values), None
            else:

                keys = torch.cat((cache_keys[..., :cache_sequence_length, :, :], incoming_keys), dim = -3)
                values = torch.cat((cache_values[..., :cache_sequence_length, :, :], incoming_values), dim = -3)

                cache_keys[..., cache_sequence_length:total_sequence_length, :, :] = incoming_keys
                cache_values[..., cache_sequence_length:total_sequence_length, :, :] = incoming_values

                # custom mask so that it can pay attention to the cached tokens
                mask = torch.ones(incoming_sequence_length, total_sequence_length, dtype = torch.bool, device = keys.device).triu()

                return (keys, values), mask


    def attention(self, activations, kv_cache, index):
        queries = self.query_layer(activations).unflatten(-1, (self.block_config.n_attn_heads, self.key_head_size))

        incoming_keys = self.key_layer(activations).unflatten(-1, (self.block_config.n_attn_heads, self.key_head_size))
        incoming_values = self.value_layer(activations).unflatten(-1, (self.block_config.n_attn_heads, self.value_head_size))

        queries, incoming_keys = self.position_embedding(queries, incoming_keys, index, index + queries.size(-3))

        (keys, values), mask = self.get_full_kv((incoming_keys, incoming_values), kv_cache, index)

        attention = torch.nn.functional.scaled_dot_product_attention(
            queries.transpose(-3, -2),
            keys.transpose(-3, -2),
            values.transpose(-3, -2),
            is_causal = (mask is None),
            dropout_p = self.network_config.dropout_rate if self.training else 0.0,
            attn_mask = mask
        # transpose to switch the sequence and head dimensions
        ).transpose(-3, -2)
        
        return self.attention_linear(attention.flatten(-2))
    

    def forward(self, activations: torch.Tensor, kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]], index) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # kv cache is modified in place
        activations = (activations + self.attention(activations, kv_cache = kv_cache, index = index)) / self.sqrt_two_constant
        activations = (activations + self.mlp(activations)) / self.sqrt_two_constant
        return activations, kv_cache




    
class transformer_network(torch.nn.Module):
    def __init__(self, config: transformer_network_config):
        super(transformer_network, self).__init__()

        self.config = config

        
        self.blocks = torch.nn.ModuleList([transformer_block(config, block_config) for block_config in config.block_configs])
        
        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)

        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        
        self.lm_head_weights = self.wte.weight

        self.sqrt_embedding_constant = torch.nn.Parameter(torch.tensor([math.sqrt(config.embedding_size)]), requires_grad=False)

    # index should start at 0
    def forward(self, encodings: torch.Tensor, kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None, index: int = 0) -> Optional[tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]]:

        embeddings = torch.nn.functional.dropout(self.wte(encodings), p = self.config.dropout_rate, training = self.training)

        for i, block in enumerate(self.blocks):
            if kv_cache is None:
                embeddings, _ = block.forward(embeddings, None, index)
            else:
                embeddings, kv_cache[i] = block.forward(embeddings, kv_cache[i], index)
        
        logits = torch.nn.functional.linear(embeddings, weight = self.lm_head_weights / self.sqrt_embedding_constant)

        return logits, kv_cache  
        
    
    def get_empty_kv_cache(self, batch_size: int, sequence_length: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return ([
            (
                torch.empty(batch_size, sequence_length, block.block_config.n_attn_heads, block.key_head_size, device = device).squeeze(-4),
                torch.empty(batch_size, sequence_length, block.block_config.n_attn_heads, block.value_head_size, device = device).squeeze(-4)
            )
        for block in self.blocks])
