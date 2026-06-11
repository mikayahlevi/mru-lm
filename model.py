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
        torch.nn.init.normal_(self.mru_out.weight, mean = 0, std = 0.02 / math.sqrt(2 * config.n_blocks))

    
    # use cayley to construct an orthogonal matrix Q, then multiply it by an upper triangular matrix R
    # by ensuring the product of the upper triangular matrix's diagonal is 1 we can ensure that QR determinant 1
    def create_state_matrix(self, state_elements: torch.Tensor) -> torch.Tensor:
        input_matrix = state_elements.unflatten(-1, (self.state_head_order, self.state_head_order))


        upper_matrix = torch.triu(input_matrix, diagonal = 1)
        
        diagonal = torch.diagonal(input_matrix, dim1 = -2, dim2 = -1)
        diagonal = torch.exp(diagonal - diagonal.mean(dim = -1, keepdim = True))

        upper_matrix[..., torch.arange(self.state_head_order), torch.arange(self.state_head_order)] = diagonal
        
        # construct a skew-symmetric matrix
        skew_symmetric_matrix = input_matrix.tril(diagonal = -1)
        skew_symmetric_matrix = skew_symmetric_matrix - skew_symmetric_matrix.transpose(-2, -1)

        # take the cayley transform
        identity = torch.eye(self.state_head_order, device = state_elements.device, dtype = state_elements.dtype)

        original_dtype = input_matrix.dtype
        with torch.amp.autocast(device_type = input_matrix.device.type, enabled = False):
            if original_dtype not in (torch.float32, torch.float64):
                skew_symmetric_matrix = skew_symmetric_matrix.float()
            
            orthogonal_matrix = torch.linalg.solve(identity - skew_symmetric_matrix, identity + skew_symmetric_matrix)
        
        orthogonal_matrix = orthogonal_matrix.to(original_dtype)

        state_matrix = orthogonal_matrix @ upper_matrix

        return state_matrix



    def forward(self, activations: torch.Tensor, last_state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # bias states by the state_matrices_base, which starts as the identity so the update rule upon initialization is simple
        input_state_elements = torch.nn.functional.dropout(
            self.state_matrices_up(activations),
            p = self.config.dropout_rate,
            training = self.training
        ).unflatten(-1, (self.config.n_state_heads, -1))

        input_states = self.create_state_matrix(input_state_elements)

        full_input_states = input_states if last_state is None else torch.cat((last_state.unsqueeze(dim = -4), input_states), dim = -4)

        # ensure that full_input_states are float32 or float64, then cast back to the original dtype
        original_dtype = full_input_states.dtype
        with torch.amp.autocast(device_type = full_input_states.device.type, enabled = False):
            if original_dtype not in (torch.float32, torch.float64):
                full_input_states = full_input_states.float()

            full_output_states = parallel_mru_op(full_input_states.transpose(-3, -4)).transpose(-3, -4)

        full_output_states = full_output_states.to(original_dtype)


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
