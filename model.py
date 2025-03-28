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



class mru_lm_block(torch.nn.Module):
    def __init__(self, config: mru_lm_config):
        super(mru_lm_block, self).__init__()

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

        self.state_matrices_up = torch.nn.Parameter(
            torch.zeros(
                config.n_state_heads, self.embedding_chunk_size, self.state_head_order
            ),
            requires_grad = True
        )

        # this scaling factor and the init for state_matrices_down is based on the μP paper
        # https://arxiv.org/abs/2412.08905
        # https://github.com/microsoft/mup
        # the scaling should make maximal update for state_matrices_up and state_matrices_down the same as the rest of the network.
        self.state_matrices_update_scale = 0.08 * (1 / self.state_head_order) * (config.embedding_size / self.embedding_chunk_size)
        self.state_matrices_down = torch.nn.Parameter(
            torch.normal(
                mean = 0,
                std = 0.02 * math.sqrt(config.state_size) / (self.config.embedding_size / self.state_head_order),
                size = (config.n_state_heads, self.state_head_order, self.embedding_chunk_size)
            ),
            requires_grad = True
        )

        self.mru_out = torch.nn.Linear(config.embedding_size, config.embedding_size, bias = False)
        torch.nn.init.normal_(self.mru_out.weight, mean = 0, std = 0.02 / math.sqrt(config.n_blocks))


        self.first_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(config.embedding_size * 4, config.embedding_size, bias = False),
            torch.nn.Dropout(config.dropout_rate)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(config.n_blocks))
            
    def mru(self, activations: torch.Tensor, last_state: Optional[torch.Tensor]) -> torch.Tensor:
        new_matrices = torch.nn.functional.dropout(
            activations.unflatten(-1, (self.config.n_state_heads, self.state_head_order, self.embedding_chunk_size)) @ self.state_matrices_up,
            p = self.config.dropout_rate,
            training = self.training
        ) * self.state_matrices_update_scale + torch.eye(self.state_head_order, device = activations.device)
        
        full_matrices = new_matrices if last_state is None else torch.cat((last_state.unsqueeze(dim = -4), new_matrices), dim = -4)
        
        parallel_mru_op_output = parallel_mru_op(full_matrices.transpose(-3, -4)).transpose(-3, -4)
        
        states = parallel_mru_op_output if last_state is None else parallel_mru_op_output[..., 1:, :, :, :]

        # multiplying by (self.config.embedding_size / self.state_head_order) is also part lr scaling
        # this allows the gradients to be larger because less inputs are being summed.
        output = ((states @ self.state_matrices_down) * (self.config.embedding_size / self.state_head_order)).flatten(-3, -1)

        return torch.nn.functional.dropout(
            self.mru_out(output),
            p = self.config.dropout_rate,
            training = self.training
        ), states[-1]


    def forward(self, activations: torch.Tensor, last_state: Optional[torch.Tensor]) -> torch.Tensor:
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
        
        self.lm_head_weights = self.wte.weight

    # index should start at 0
    def forward(self, encodings: torch.Tensor, last_state: Optional[list[Optional[torch.Tensor]]] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        embeddings = torch.nn.functional.dropout( 
            self.wte(encodings),
            p = self.config.dropout_rate,
            training = self.training
        )

        if last_state is None:
            last_state = self.get_initial_state()

        for i, block in enumerate(self.blocks):
            embeddings, last_state[i] = block.forward(embeddings, last_state[i])
        
        logits = torch.nn.functional.linear(embeddings, weight = self.lm_head_weights)

        return logits, last_state
        
    
    def get_initial_state(self) -> list[torch.Tensor]:
        return ([None for _ in self.blocks])
