import torch
import datasets

import time
import math
import colorama
import os
import math

from dataclasses import dataclass
from typing import Callable, Any, cast, Optional

from pipeline import pipeline_protocol

@dataclass
class train_config:
    sequence_length: int
    batch_size: int

    shuffle: bool

    total_steps: int

    update_interval: int

    schedule_interval: int



@dataclass
class hyperparameter_config:
    lr_warmup_steps: int
    lr_decay_steps: int

    start_lr: float
    peak_lr: float
    end_lr: float

    betas: tuple[float, float]

    weight_decay: float


def configure_optimizer(model, hyperparameters, sequence_length: int):
    wte_weight = model.wte.weight

    final_ln_weight = model.final_ln.weight

    first_ln_weights = [block.first_ln.weight for block in model.blocks]
    second_ln_weights = [block.second_ln.weight for block in model.blocks]

    mlp_up_weights = [block.mlp[0].weight for block in model.blocks]
    mlp_down_weights = [block.mlp[2].weight for block in model.blocks]

    state_matrices_up_weights = [block.mru.state_matrices_up.weight for block in model.get_mru_blocks()]
    state_matrices_down_weights = [block.mru.state_matrices_down for block in model.get_mru_blocks()]
    mru_out_weights = [block.mru.mru_out.weight for block in model.get_mru_blocks()]

    query_layer_weights = [block.attn.query_layer.weight for block in model.get_attn_blocks()]
    key_layer_weights = [block.attn.key_layer.weight for block in model.get_attn_blocks()]
    value_layer_weights = [block.attn.value_layer.weight for block in model.get_attn_blocks()]
    attention_down_weights = [block.attn.attention_down.weight for block in model.get_attn_blocks()]


    state_head_order = math.isqrt(model.config.state_size // model.config.n_state_heads)

    # during attention each token affects approximately one other token
    # the MRU on average affects all future tokens therefore recieves much more gradients information
    # therefore scale the down update's variance by sequence_length / 2; the average number of future tokens

    # furthermore, the state matrices have fan_in of state_head_order
    # based on μP, we should scale the lr by 1 / state_head_order
    state_matrices_update_scale = math.sqrt(1 / state_head_order) * math.sqrt(2 / sequence_length)

    optim_groups = [
        {
            'params': [wte_weight] + [final_ln_weight] + first_ln_weights + second_ln_weights,
            'weight_decay': 0.0,
            'max_grad_norm': 1.0
        },
        {
            'params': mlp_up_weights + mlp_down_weights + state_matrices_down_weights + mru_out_weights + query_layer_weights + key_layer_weights + value_layer_weights + attention_down_weights,
            'weight_decay': hyperparameters.weight_decay,
            'max_grad_norm': 1.0
        },
        {
            'params': state_matrices_up_weights,
            'weight_decay': 0.0,
            'lr': state_matrices_update_scale,
            'max_grad_norm': math.sqrt(sequence_length / 2)
        }
    ]

    return torch.optim.AdamW(optim_groups, lr = 1.0, betas = hyperparameters.betas)



def format_info(info):
    tmp = ''
    for i, (key, value) in enumerate(info.items()):
        if i == 0:
            tmp += str(value).rjust(6) + ' ' + key
        else:
            tmp += '      ' + str(value).rjust(6) + ' ' + key

    return tmp


class accumulating_metric:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float):
        self.sum += value
        self.count += 1

    def get_average_and_reset(self) -> float:
        avg = self.sum / self.count if self.count > 0 else 0.0
        self.sum = 0.0
        self.count = 0

        return avg



def train(
    settings: train_config,
    hyperparameters: hyperparameter_config,

    model,

    checkpoint,

    dataset: datasets.DatasetDict,

    tokenizer,

    pipeline: pipeline_protocol[Any, Any],

    log_metric: Callable[[int, str, float], None],

    loss_log_interval: int,
    eval_interval: int,
    metric_print_interval: int,
    checkpoint_save_interval: int,

    checkpoint_save_path: str,

    device,

    precision: Optional[str] = None
):
    print(colorama.Fore.GREEN)
    print('starting training')
    print(colorama.Style.RESET_ALL, end='')

    device_type = torch.device(device).type
    mp_dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
        None: torch.float32
    }[precision]

    mixed_precision = mp_dtype != torch.float32


    train_dataloader, val_dataloader = pipeline.get_dataloaders(dataset, batch_size = settings.batch_size, shuffle = settings.shuffle, pin_memory = (device == 'cuda'))
    train_dataloader_iter = iter(train_dataloader)

    mask_value = -100

    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean', ignore_index = mask_value)


    def lr_lambda(sched_step: int) -> float:
        train_step = sched_step * settings.schedule_interval
        decay_step = train_step - hyperparameters.lr_warmup_steps

        if train_step < hyperparameters.lr_warmup_steps:
            scale = train_step / hyperparameters.lr_warmup_steps
            return hyperparameters.start_lr + scale * (hyperparameters.peak_lr - hyperparameters.start_lr)
        elif decay_step < hyperparameters.lr_decay_steps:
            # interpolates between 1 and 0
            scale = 0.5 * (1 + math.cos(math.pi * decay_step / hyperparameters.lr_decay_steps))
            return hyperparameters.end_lr + scale * (hyperparameters.peak_lr - hyperparameters.end_lr)
        else:
            return hyperparameters.end_lr

    optimizer = configure_optimizer(model, hyperparameters, settings.sequence_length)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lr_lambda
    )

    # scaling not needed with bf16
    scaler = torch.amp.GradScaler(device = device_type, enabled = (mp_dtype == torch.float16))

    start_step = 0
    if checkpoint is not None:
        # note the model state dict is already loaded
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']

        print(colorama.Fore.YELLOW)
        print('resuming training from checkpoint at step', start_step)
        print(colorama.Style.RESET_ALL, end='')

        # advance dataloader to the correct position within the current epoch
        for _ in range(start_step % len(train_dataloader)):
            next(train_dataloader_iter)



    logged_tr_loss_avg = accumulating_metric()
    display_tr_loss_sum = accumulating_metric()
    last_val_loss = 'n/a'


    start = time.time()
    for step in range(start_step, settings.total_steps):
        model.train()

        ids = next(train_dataloader_iter)['ids']

        inputs, labels = pipeline.get_training_pairs(ids, tokenizer,  mask_value)
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast(device_type = device_type, dtype = mp_dtype, enabled = mixed_precision):
            logits, _ = model(inputs)

            # flatten batch and sequence dimensions into one dimension for computing the loss
            loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))

        if loss.isnan():
            print(colorama.Fore.RED)
            print('quitting: NaN loss encountered at step', step + 1)
            print(colorama.Style.RESET_ALL, end='')
            break


        logged_tr_loss_avg.update(loss.item())
        display_tr_loss_sum.update(loss.item())


        scaler.scale(loss / settings.update_interval).backward()



        completed_steps = step + 1

        if completed_steps % len(train_dataloader) == 0:
            # reset the dataloader iterator at the end of each epoch
            train_dataloader_iter = iter(train_dataloader)

        if completed_steps % settings.update_interval == 0:
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        if completed_steps % settings.schedule_interval == 0:
            scheduler.step()

            log_metric(completed_steps, 'lr', float(scheduler.get_last_lr()[0]))


        if completed_steps % loss_log_interval == 0:
            log_metric(completed_steps, 'train_loss', logged_tr_loss_avg.get_average_and_reset())


        if completed_steps % eval_interval == 0:
            val_loss_acc = accumulating_metric()

            model.eval()
            with torch.inference_mode():
                for val_item in iter(val_dataloader):
                    val_inputs, val_labels = pipeline.get_training_pairs(val_item['ids'], tokenizer, mask_value)
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    with torch.amp.autocast(device_type = device_type, dtype = mp_dtype, enabled = mixed_precision):
                        val_logits, _ = model(val_inputs)

                        val_loss = criterion(val_logits.flatten(-3, -2), val_labels.flatten(-2, -1))

                    val_loss_acc.update(val_loss.item())

            model.train()

            last_val_loss = val_loss_acc.get_average_and_reset()

            log_metric(completed_steps, 'val_loss', last_val_loss)


        if completed_steps % metric_print_interval == 0:
            print_info = {
                'time elapsed': time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                'done': '{:.2f}'.format(100 * completed_steps / settings.total_steps) + '%',
                'steps': completed_steps,
                'epochs': completed_steps // len(train_dataloader),
            }


            print_info['avg train loss'] = '{:.4f}'.format(display_tr_loss_sum.get_average_and_reset())

            print_info['last val loss'] = '{:.4f}'.format(last_val_loss) if last_val_loss != 'n/a' else 'n/a'

            print_info['current lr'] = '{:.6f}'.format(scheduler.get_last_lr()[0])

            # print the info
            print(colorama.Fore.YELLOW, end='')
            print(format_info(print_info))
            print(colorama.Style.RESET_ALL, end='')


        # this should remain last so that everything is updated before saving the checkpoint
        if completed_steps % checkpoint_save_interval == 0:
            torch.save({
                'step': completed_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
            }, os.path.join(checkpoint_save_path, 'checkpoint-' + 'step-' + str(completed_steps) + '.pt'))

    print(colorama.Fore.GREEN)
    print('training finished:', settings.total_steps, 'steps completed')
    print(colorama.Style.RESET_ALL, end='')
