import torch
import math


# hs: Hillis-Steele scan
class hs_parallel_mru_op_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, start_matrix_states):
        final_matrix_states = start_matrix_states.clone()

        sequence_length = start_matrix_states.size(-3)
        
        n_stages = math.ceil(math.log2(sequence_length))
        for stage in range(n_stages):
            stage_stride = 2 ** stage
            final_matrix_states[..., stage_stride:, :, :] = final_matrix_states[..., :-stage_stride, :, :] @ final_matrix_states[..., stage_stride:, :, :]
        

        ctx.save_for_backward(start_matrix_states, final_matrix_states)
        ctx.sequence_length = sequence_length

        return final_matrix_states

    @staticmethod
    def backward(ctx, grad_final_matrix_states):
        def create_eye_for_shift(shape):
            resized_eye = torch.eye(*shape[-2:], device = grad_final_matrix_states.device)
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
            return torch.zeros(new_shape, device = grad_final_matrix_states.device)
        
        start_matrix_states, final_matrix_states = ctx.saved_tensors

        # grad_before_start_matrix_states is A as described in the README
        # tl is U as described in the README
        # bl is L as described in the README


        # grad_before_start_matrix_states = torch.cat((create_eye_for_shift(transposed_final_matrix_states.shape), transposed_final_matrix_states[..., :-1, :, :]), dim = -3)
        
        # faster implementation:
        grad_before_start_matrix_states = final_matrix_states.transpose(-1, -2).roll(1, dims = -3)
        grad_before_start_matrix_states[..., 0, :, :] = torch.eye(grad_before_start_matrix_states.size(-2), device = grad_before_start_matrix_states.device)


        # tl = torch.cat((start_matrix_states[..., 1:, :, :], create_zeros_for_shift(start_matrix_states.shape)), dim = -3).transpose(-1, -2)
        
        # faster implementation:
        tl = start_matrix_states.transpose(-1, -2).roll(-1, dims = -3)
        tl[..., -1, :, :] = torch.zeros((tl.size(-2), tl.size(-1)), device = tl.device)

        bl = grad_final_matrix_states

        sequence_length = ctx.sequence_length
        n_stages = math.ceil(math.log2(sequence_length))
        for stage in range(n_stages):
            stage_stride = 2 ** stage
            bl[..., :-stage_stride, :, :] = bl[..., stage_stride:, :, :] @ tl[..., :-stage_stride, :, :] + bl[..., :-stage_stride, :, :]
            tl[..., :-stage_stride, :, :] = tl[..., stage_stride:, :, :] @ tl[..., :-stage_stride, :, :]

        grad_start_matrix_states = grad_before_start_matrix_states @ bl

        return grad_start_matrix_states
    
op = hs_parallel_mru_op_class.apply