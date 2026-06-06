import torch

import math

from typing import Optional
from dataclasses import dataclass

# set to mru_scans.bk_mru, mru_scans.hs_mru, or mru_scans.cuda_mru
# cuda_mru is the fastest but is still a work in progress and therefore only supports certain tensor shapes
from mru_scans.cuda_mru import op as parallel_mru_op

@dataclass
class mru_lm_config:
    vocab_size: int


    embedding_size: int

    dropout_rate: float

    n_state_heads: int
    state_size: int

    n_blocks: int



class mru(torch.nn.Module):
    def __init__(self, config: mru_lm_config):
        super(mru, self).__init__()

        self.config = config

        if config.state_size % config.n_state_heads != 0:
            raise ValueError("state size must be divisible by the number of state heads")
        self.state_head_size = config.state_size // config.n_state_heads

        if self.state_head_size != math.isqrt(self.state_head_size) ** 2:
            raise ValueError("state head size must be a peffect square to form the state head matrix")
        self.state_head_order = math.isqrt(self.state_head_size)

        if config.embedding_size % self.state_head_order != 0:
            raise ValueError(f"embedding size must be divisible by the state head order ({self.state_head_order})")
        self.embedding_chunk_size = config.embedding_size // (self.state_head_order * config.n_state_heads)

        self.state_matrices_up = torch.nn.Linear(config.embedding_size, config.state_size, bias = False)
        torch.nn.init.zeros_(self.state_matrices_up.weight)

        # scaling factor based on μP
        self.state_matrices_update_scale = (1 / self.state_head_order)


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
        torch.nn.init.normal_(self.mru_out.weight, mean = 0, std = 0.02 / math.sqrt(2 * config.n_blocks))

    def forward(self, activations: torch.Tensor, last_state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # bias states by the identity so the update rule upon initialization is simple
        input_states = torch.nn.functional.dropout(
            self.state_matrices_up(activations).unflatten(-1, (self.config.n_state_heads, self.state_head_order, self.state_head_order)),
            p = self.config.dropout_rate,
            training = self.training
        ) * self.state_matrices_update_scale + torch.eye(self.state_head_order, device = activations.device)

        full_input_states = input_states if last_state is None else torch.cat((last_state.unsqueeze(dim = -4), input_states), dim = -4)

        full_output_states = parallel_mru_op(full_input_states.transpose(-3, -4)).transpose(-3, -4)

        output_states = full_output_states if last_state is None else full_output_states[..., 1:, :, :, :]

        output = (output_states @ self.state_matrices_down).flatten(-3, -1)

        return torch.nn.functional.dropout(
            self.mru_out(output),
            p = self.config.dropout_rate,
            training = self.training
        ), output_states[..., -1, :, :, :]


class mru_lm_block(torch.nn.Module):
    def __init__(self, config: mru_lm_config):
        super(mru_lm_block, self).__init__()

        self.config = config


        self.first_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(config.embedding_size * 4, config.embedding_size, bias = False),
            torch.nn.Dropout(config.dropout_rate)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(2 * config.n_blocks))

        self.mru = mru(config)

    def forward(self, activations: torch.Tensor, last_state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mru_out, new_state = self.mru(self.first_ln(activations), last_state)

        activations = activations + mru_out
        activations = activations + self.mlp(self.second_ln(activations))
        return activations, new_state




class mru_lm_network(torch.nn.Module):
    def __init__(self, config: mru_lm_config):
        super(mru_lm_network, self).__init__()

        self.config = config

        self.blocks = torch.nn.ModuleList([mru_lm_block(config) for _ in range(config.n_blocks)])

        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)

        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)

        self.lm_head = torch.nn.Linear(config.embedding_size, config.vocab_size, bias = False)
        self.lm_head.weight = self.wte.weight

    def forward(self, encodings: torch.Tensor, last_state: Optional[list[torch.Tensor]] | list[None] = None) -> tuple[torch.Tensor, list[torch.Tensor] | list[None]]:
        embeddings = torch.nn.functional.dropout(
            self.wte(encodings),
            p = self.config.dropout_rate,
            training = self.training
        )

        if last_state is None:
            last_state = self.get_initial_state()

        for i, block in enumerate(self.blocks):
            embeddings, last_state[i] = block.forward(embeddings, last_state[i])

        embeddings = self.final_ln(embeddings)
        logits = self.lm_head(embeddings)

        return logits, last_state

    def get_initial_state(self) -> list[torch.Tensor] | list[None]:
        return ([None for _ in self.blocks])
