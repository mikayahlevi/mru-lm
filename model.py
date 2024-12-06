import torch

import math

from typing import Optional
from dataclasses import dataclass

@dataclass
class mrun_config:
    vocab_size: int
    
    embedding_size: int

    dropout_rate: float

    n_state_heads: int
    state_size: int

    n_blocks: int



class parallel_mru_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_state, start_matrix_states):
        final_matrix_states = start_matrix_states.clone()

        sequence_length = start_matrix_states.size(-3)
        
        n_stages = math.ceil(math.log2(sequence_length))
        for stage in range(n_stages):
            stage_stride = 2 ** stage
            final_matrix_states[..., stage_stride:, :, :] = final_matrix_states[..., :-stage_stride, :, :] @ final_matrix_states[..., stage_stride:, :, :]
        

        ctx.save_for_backward(input_state, start_matrix_states, final_matrix_states)
        ctx.sequence_length = sequence_length

        return (input_state.unsqueeze(-2).unsqueeze(-2) @ final_matrix_states).squeeze(-2)

    @staticmethod
    def backward(ctx, grad_output):
        def create_eye_for_shift(shape):
            resized_eye = torch.eye(*shape[-2:], device = grad_output.device)
            while resized_eye.dim() < len(shape):
                resized_eye = resized_eye.unsqueeze(0)
            
            resized_eye_shape = shape[:-3]
            resized_eye_shape = list(resized_eye_shape)
            
            while len(resized_eye_shape) < len(shape):
                resized_eye_shape.append(1)

            resized_eye = resized_eye.repeat(*resized_eye_shape)
            return resized_eye

        def create_zeros_for_shift(shape):
            new_shape = list(shape)
            new_shape[-3] = 1
            return torch.zeros(new_shape, device = grad_output.device)
        
        input_state, start_matrix_states, final_matrix_states = ctx.saved_tensors

        transposed_final_matrix_states = final_matrix_states.transpose(-1, -2)

        grad_input_state = (grad_output.unsqueeze(-2) @ transposed_final_matrix_states).sum(dim = -3).squeeze(-2)
        grad_final_matrix_states = input_state.unsqueeze(-1).unsqueeze(-3) * grad_output.unsqueeze(-2)


        # grad_before_start_matrix_states = torch.cat((create_eye_for_shift(transposed_final_matrix_states.shape), transposed_final_matrix_states[..., :-1, :, :]), dim = -3)
        # faster implementation

        grad_before_start_matrix_states = transposed_final_matrix_states.roll(1, dims = -3)
        grad_before_start_matrix_states[..., 0, :, :] = torch.eye(grad_before_start_matrix_states.size(-2), device = grad_before_start_matrix_states.device)


        # tl = torch.cat((start_matrix_states[..., 1:, :, :], create_zeros_for_shift(start_matrix_states.shape)), dim = -3).transpose(-1, -2)
        # faster implementation

        tl = start_matrix_states.roll(-1, dims = -3).transpose(-1, -2)
        tl[..., -1, :, :] = torch.zeros((tl.size(-2), tl.size(-1)), device = tl.device)

        bl = grad_final_matrix_states

        sequence_length = ctx.sequence_length
        n_stages = math.ceil(math.log2(sequence_length))
        for stage in range(n_stages):
            stage_stride = 2 ** stage
            bl[..., :-stage_stride, :, :] = bl[..., stage_stride:, :, :] @ tl[..., :-stage_stride, :, :] + bl[..., :-stage_stride, :, :]
            tl[..., :-stage_stride, :, :] = tl[..., stage_stride:, :, :] @ tl[..., :-stage_stride, :, :]

        grad_start_matrix_states = grad_before_start_matrix_states @ bl

        return grad_input_state, grad_start_matrix_states

    
class genmatrix_module(torch.nn.Module):
    def __init__(self, input_size, resolution, n_state_heads, state_head_size, lr_like = 0.003):
        super(genmatrix_module, self).__init__()

        self.resolution = resolution
        self.input_size = input_size
        self.n_state_heads = n_state_heads
        self.state_head_size = state_head_size
        self.lr_like = lr_like

        self.query_layer = torch.nn.Linear(input_size, resolution * n_state_heads * state_head_size, bias = False)
        self.value_layer = torch.nn.Linear(input_size, resolution * n_state_heads * state_head_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 1.0 / math.sqrt(input_size))
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 1.0 / math.sqrt(input_size))
        # torch.nn.init.zeros_(self.query_layer.weight)
        # torch.nn.init.zeros_(self.value_layer.weight)

        self.eye = torch.nn.Parameter(torch.eye(state_head_size), requires_grad=False)

    def forward(self, input):
        queries = self.query_layer(input).unflatten(-1, (self.resolution, self.n_state_heads, self.state_head_size))
        values = self.value_layer(input).unflatten(-1, (self.resolution, self.n_state_heads, self.state_head_size))

        matrices = (queries.unsqueeze(-1) @ values.unsqueeze(-2))

        return self.eye + (matrices.sum(dim = -4).transpose(-3, -4) * self.lr_like)



class mrun_block(torch.nn.Module):
    def __init__(self, config: mrun_config):
        super(mrun_block, self).__init__()

        self.config = config


        if config.state_size % config.n_state_heads != 0:
            raise ValueError("state size must be divisible by the number of state heads")
        self.state_head_size = config.state_size // config.n_state_heads


        self.genmatrix = genmatrix_module(config.embedding_size, 4, config.n_state_heads, self.state_head_size)

        self.state_down = torch.nn.Linear(config.state_size, config.embedding_size, bias = False)
        torch.nn.init.normal_(self.state_down.weight, mean = 0, std = (0.02 * 0.4) / math.sqrt(config.n_blocks))


        self.first_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(config.embedding_size * 4, config.embedding_size, bias = False)
            torch.nn.Dropout(config.dropout_rate)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(config.n_blocks))
        
        self.initial_state = torch.nn.Parameter(torch.normal(mean = 0, std = 1, size = (config.n_state_heads, self.state_head_size)), requires_grad=True)
    
    def parallel_mru(self, activations: torch.Tensor, last_state: torch.Tensor) -> torch.Tensor:
        matrices = self.genmatrix(activations)
        return parallel_mru_class.apply(last_state, matrices).transpose(-2, -3)
    
    def process_states(self, states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.dropout(
            self.state_down(states.flatten(-2, -1)),
            p = self.config.dropout_rate,
            training = self.training
        )


    def forward(self, activations: torch.Tensor, last_state: torch.Tensor) -> torch.Tensor:
        states = self.parallel_mru(self.first_ln(activations), last_state)

        activations = activations + self.process_states(states)
        activations = activations + self.mlp(self.second_ln(activations))
        return activations, states[-1]








    
class mrun_network(torch.nn.Module):
    def __init__(self, config: mrun_config):
        super(mrun_network, self).__init__()

        self.config = config

        self.blocks = torch.nn.ModuleList([mrun_block(config) for _ in range(config.n_blocks)])
        
        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)
        
        self.lm_head_weights = self.wte.weight

    # index should start at 0
    def forward(self, encodings: torch.Tensor, last_state: list[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        embeddings = torch.nn.functional.dropout( 
            self.wte(encodings),
            p = self.config.dropout_rate,
            training = self.training
        )

        for i, block in enumerate(self.blocks):
            embeddings, last_state[i] = block.forward(embeddings, last_state[i])
        
        logits = torch.nn.functional.linear(embeddings, weight = self.lm_head_weights)

        return logits, last_state
        
    
    def get_initial_state(self) -> list[torch.Tensor]:
        return ([block.initial_state for block in self.blocks])
