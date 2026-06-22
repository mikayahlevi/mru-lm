import torch

import math

from typing import Optional
from dataclasses import dataclass

# set to mru_scans.bk_mru, mru_scans.hs_mru, or mru_scans.cuda_mru
# cuda_mru is the fastest but is still a work in progress and therefore only supports certain tensor shapes
from mru_scans.cuda_mru import op as parallel_mru_op

@dataclass
class hybrid_lm_config:
    vocab_size: int


    embedding_size: int

    dropout_rate: float

    n_state_heads: int
    state_size: int


    n_attn_heads: int

    key_size: int
    value_size: int

    # processed through layer_name_map
    layers: list[str]


    max_sequence_length: int



class mru(torch.nn.Module):
    def __init__(self, config: hybrid_lm_config):
        super(mru, self).__init__()

        self.config = config

        if config.state_size % config.n_state_heads != 0:
            raise ValueError("state size must be divisible by the number of state heads")
        self.state_head_size = config.state_size // config.n_state_heads

        if self.state_head_size != math.isqrt(self.state_head_size) ** 2:
            raise ValueError("state head size must be a perfect square to form the state head matrix")
        self.state_head_order = math.isqrt(self.state_head_size)

        if config.embedding_size % self.state_head_order != 0:
            raise ValueError(f"embedding size must be divisible by the state head order ({self.state_head_order})")
        self.embedding_chunk_size = config.embedding_size // (self.state_head_order * config.n_state_heads)

        n_state_elements = config.state_size
        # n_state_elements = config.n_state_heads * (self.state_head_order * (self.state_head_order - 1)) // 2

        self.state_matrices_up = torch.nn.Linear(config.embedding_size, n_state_elements, bias = False)
        torch.nn.init.zeros_(self.state_matrices_up.weight)

        self.state_matrices_down = torch.nn.Parameter(
            torch.normal(
                mean = 0,
                # upon initialization, the output_states will have the variance of a flattened identity matrix
                # which is 1 / order, so scale it up as if it was normalized to std 1.
                std = 0.02 * math.sqrt(self.state_head_order),
                size = (config.n_state_heads, self.state_head_order, self.embedding_chunk_size)
            ),
            requires_grad = True
        )

        self.mru_out = torch.nn.Linear(config.embedding_size, config.embedding_size, bias = False)
        torch.nn.init.normal_(self.mru_out.weight, mean = 0, std = 0.02 / math.sqrt(len(config.layers)))


    def create_state_matrix(self, state_elements: torch.Tensor) -> torch.Tensor:
        input_matrix = state_elements.unflatten(-1, (self.state_head_order, self.state_head_order))

        lower_matrix = input_matrix.tril(diagonal = -1)
        upper_matrix = input_matrix.triu(diagonal = 1)

        diagonal = torch.diagonal(input_matrix, dim1 = -2, dim2 = -1)
        diagonal = torch.exp(diagonal - diagonal.mean(dim = -1, keepdim = True)).sqrt()

        diagonal_matrix = torch.zeros_like(input_matrix)
        diagonal_matrix.diagonal(dim1 = -2, dim2 = -1).copy_(diagonal)

        return (lower_matrix + diagonal_matrix) @ (upper_matrix + diagonal_matrix)



    def forward(self, activations: torch.Tensor, prev_state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # bias states by the state_matrices_base, which starts as the identity so the update rule upon initialization is simple
        input_state_elements = torch.nn.functional.dropout(
            self.state_matrices_up(activations),
            p = self.config.dropout_rate,
            training = self.training
        ).unflatten(-1, (self.config.n_state_heads, -1))

        input_states = self.create_state_matrix(input_state_elements)

        full_input_states = input_states if prev_state is None else torch.cat((prev_state.unsqueeze(dim = -4), input_states), dim = -4)

        # ensure that full_input_states are float32 or float64, then cast back to the original dtype
        original_dtype = full_input_states.dtype
        with torch.amp.autocast(device_type = full_input_states.device.type, enabled = False):
            if original_dtype not in (torch.float32, torch.float64):
                full_input_states = full_input_states.float()

            full_output_states = parallel_mru_op(full_input_states.transpose(-3, -4)).transpose(-3, -4)

        full_output_states = full_output_states.to(original_dtype)


        output_states = full_output_states if prev_state is None else full_output_states[..., 1:, :, :, :]

        output = (output_states @ self.state_matrices_down).flatten(-3, -1)

        return torch.nn.functional.dropout(
            self.mru_out(output),
            p = self.config.dropout_rate,
            training = self.training
        ), output_states[..., -1, :, :, :]



class attention_cache(torch.nn.Module):
    def __init__(self, config: hybrid_lm_config, preceeding_dimensions: tuple[int, ...] = ()):
        super(attention_cache, self).__init__()

        self.prev_position = 0
        self.curr_position = 0

        n_attn_layers = config.layers.count("attn")

        self.keys = torch.nn.Buffer(torch.empty(
            preceeding_dimensions + (n_attn_layers, config.max_sequence_length, config.n_attn_heads, config.key_size // config.n_attn_heads)
        ))
        self.values = torch.nn.Buffer(torch.empty(
            preceeding_dimensions + (n_attn_layers, config.max_sequence_length, config.n_attn_heads, config.value_size // config.n_attn_heads)
        ))

    def increment_position(self, amount: int):
        self.prev_position = self.curr_position
        self.curr_position += amount

    def reset(self):
        self.prev_position = 0
        self.curr_position = 0

    def append_keys(self, keys: torch.Tensor, attn_index: int):
        self.keys[..., attn_index, self.prev_position:self.curr_position, :, :] = keys

    def append_values(self, values: torch.Tensor, attn_index: int):
        self.values[..., attn_index, self.prev_position:self.curr_position, :, :] = values

    def get_full_keys(self, attn_index: int) -> torch.Tensor:
        return self.keys[..., attn_index, :self.curr_position, :, :]

    def get_full_values(self, attn_index: int) -> torch.Tensor:
        return self.values[..., attn_index, :self.curr_position, :, :]

    def get_previous_keys(self, attn_index: int) -> torch.Tensor:
        return self.keys[..., attn_index, :self.prev_position, :, :]

    def get_previous_values(self, attn_index: int) -> torch.Tensor:
        return self.values[..., attn_index, :self.prev_position, :, :]


    def get_mask(self) -> torch.Tensor:
        return torch.ones(
            (self.curr_position - self.prev_position, self.curr_position), dtype = torch.bool, device = self.keys.device
        ).tril(self.prev_position)



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

        self.c = torch.nn.Buffer(torch.cos(seq_range * theta.view(1, 1, -1)))
        self.s = torch.nn.Buffer(torch.sin(seq_range * theta.view(1, 1, -1)))
        self.t = torch.nn.Buffer((zeta.view(1, 1, -1) ** seq_range))
        self.invt = torch.nn.Buffer(1 / self.t)

    def rotate_every_two(self, input: torch.Tensor) -> torch.Tensor:
        return torch.stack((-input[..., 1::2], input[..., 0::2]), dim = -1).flatten(-2)

    def forward(self, queries, keys, start, end) -> tuple[torch.Tensor, torch.Tensor]:
        queries = (queries * self.c[start:end] + self.rotate_every_two(queries) * self.s[start:end]) * self.t[start:end]
        keys = (keys * self.c[start:end] + self.rotate_every_two(keys) * self.s[start:end]) * self.invt[start:end]


        return queries, keys



class attention(torch.nn.Module):
    def __init__(self, config):
        super(attention, self).__init__()

        self.config = config

        if config.key_size % config.n_attn_heads != 0:
            raise ValueError("key size must be divisible by the number of attention heads")
        self.key_head_size = config.key_size // config.n_attn_heads

        if config.value_size % config.n_attn_heads != 0:
            raise ValueError("value size must be divisible by the number of attention heads")
        self.value_head_size = config.value_size // config.n_attn_heads


        self.query_layer = torch.nn.Linear(config.embedding_size, config.key_size, bias = False)
        self.key_layer = torch.nn.Linear(config.embedding_size, config.key_size, bias = False)
        self.value_layer = torch.nn.Linear(config.embedding_size, config.value_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 0.02)

        self.attention_down = torch.nn.Linear(config.value_size, config.embedding_size, bias = False)
        torch.nn.init.normal_(self.attention_down.weight, mean = 0, std = 0.02 / math.sqrt(len(config.layers)))


    def forward(self, activations: torch.Tensor, process_qkv) -> torch.Tensor:
        queries = self.query_layer(activations).unflatten(-1, (self.config.n_attn_heads, self.key_head_size))

        keys = self.key_layer(activations).unflatten(-1, (self.config.n_attn_heads, self.key_head_size))
        values = self.value_layer(activations).unflatten(-1, (self.config.n_attn_heads, self.value_head_size))

        # process_qkv applies position embedding, appends keys and values to cache, and retrieves full keys and values
        queries, keys, values, mask = process_qkv(queries, keys, values)

        # transpose to switch the sequence and head dimensions
        output = torch.nn.functional.scaled_dot_product_attention(
            queries.transpose(-3, -2),
            keys.transpose(-3, -2),
            values.transpose(-3, -2),
            is_causal = mask is None,
            dropout_p = self.config.dropout_rate if self.training else 0.0,
            attn_mask = mask
        ).transpose(-3, -2)


        return torch.nn.functional.dropout(
            self.attention_down(
                output.flatten(-2)
            ),
            p = self.config.dropout_rate,
            training = self.training
        )



class mlp(torch.nn.Module):
    def __init__(self, config: hybrid_lm_config):
        super(mlp, self).__init__()

        self.config = config

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(config.embedding_size * 4, config.embedding_size, bias = False),
            torch.nn.Dropout(config.dropout_rate)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(len(config.layers)))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.mlp(activations)


layer_name_map: dict[str, type] = {
    'mru': mru,
    'attn': attention,
    'mlp': mlp
}


class hybrid_lm_network(torch.nn.Module):
    def __init__(self, config: hybrid_lm_config):
        super(hybrid_lm_network, self).__init__()

        self.config = config

        layer_class_map = {v: k for k, v in layer_name_map.items()}

        self.layers = torch.nn.ModuleList([
            layer_name_map[layer_name](config) for layer_name in config.layers
        ])

        self.specific_layer_indices = [self.config.layers[:i].count(layer_class_map[type(layer)]) for i, layer in enumerate(self.layers)]

        self.pre_lns = torch.nn.ModuleList([
            torch.nn.LayerNorm(config.embedding_size, bias = False) for _ in config.layers
        ])

        self.position_embedding = xpos(config.key_size // config.n_attn_heads, max_sequence_length = config.max_sequence_length)

        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)

        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)

        self.lm_head = torch.nn.Linear(config.embedding_size, config.vocab_size, bias = False)
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        encodings: torch.Tensor,
        prev_state: Optional[list[torch.Tensor]] = None,
        attn_cache: Optional[attention_cache] = None
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        embeddings = torch.nn.functional.dropout(
            self.wte(encodings),
            p = self.config.dropout_rate,
            training = self.training
        )

        if attn_cache is not None:
            attn_cache.increment_position(embeddings.size(-2))

        new_state = []

        for index, (layer, pre_ln) in enumerate(zip(self.layers, self.pre_lns)):
            embeddings = pre_ln(embeddings)

            # count of the previous layers of the same type
            specific_index = self.specific_layer_indices[index]

            if isinstance(layer, mru):
                this_prev_state = None
                if prev_state is not None:
                    this_prev_state = prev_state[specific_index]

                layer_out, this_new_state = layer(embeddings, this_prev_state)

                new_state.append(this_new_state)
            elif isinstance(layer, attention):
                def process_qkv(q, k, v):
                    mask = None

                    q, k = self.position_embedding(
                        q,
                        k,
                        0 if attn_cache is None else attn_cache.prev_position,
                        layer_out.size(-2) if attn_cache is None else attn_cache.curr_position
                    )

                    if attn_cache is not None:
                        attn_cache.append_keys(k, specific_index)
                        attn_cache.append_values(v, specific_index)

                        k = attn_cache.get_full_keys(specific_index)
                        v = attn_cache.get_full_values(specific_index)

                        mask = attn_cache.get_mask()

                    return q, k, v, mask


                layer_out = layer(embeddings, process_qkv)
            elif isinstance(layer, mlp):
                layer_out = layer(embeddings)



            embeddings = embeddings + layer_out


        embeddings = self.final_ln(embeddings)
        logits = self.lm_head(embeddings)

        return logits, new_state

    def get_attn_layers(self) -> list[attention]:
        return [layer for layer in self.layers if isinstance(layer, attention)]

    def get_mru_layers(self) -> list[mru]:
        return [layer for layer in self.layers if isinstance(layer, mru)]

    def get_mlp_layers(self) -> list[mlp]:
        return [layer for layer in self.layers if isinstance(layer, mlp)]
