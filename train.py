import torch
import numpy as np

import time
import colorama
import os


from dataclasses import dataclass
from matplotlib import pyplot
from typing import Optional

from model import mru_lm_config


@dataclass
class train_config:
    sequence_length: int
    batch_size: int


    total_steps: int = 5000


    update_every: int = 1

    schedule_every: int = 20

    log_loss_every: int = 20
    # eval on the validation dataset
    eval_every: int = 200


    print_progress_every: int = 40

    
    save_checkpoint_every: int = 1000

    train_log_path: Optional[str] = None
    
    


@dataclass
class hyperparameter_config:
    lr_warmup_steps: int
    lr_decay_steps: int
    
    start_lr: float
    peak_lr: float
    end_lr: float

    betas: tuple[float, float]

    weight_decay: float


def get_params_with_names(named_parameters, names: list[str]):
    filtered_parameters = []
    for n, p in named_parameters:
        for name in names:
            if name in n:
                filtered_parameters.append(p)
                break
    return filtered_parameters

    
def remove_params_with_names(named_parameters, names: list[str]):
    filtered_parameters = []
    for n, p in named_parameters:
        name_in_n = False
        for name in names:
            if name in n:
                name_in_n = True
        if name_in_n == False:
            filtered_parameters.append(p)
    return filtered_parameters


def configure_optimizer(model, hyperparameters):
    nodecay_param_names = ['ln.weight', 'wte.weight']

    optim_groups = [
        {
            'params': remove_params_with_names(model.named_parameters(), nodecay_param_names),
            'weight_decay': hyperparameters.weight_decay
        },
        {
            'params': get_params_with_names(model.named_parameters(), nodecay_param_names),
            'weight_decay': 0.0
        }
    ]

    return torch.optim.AdamW(optim_groups, lr = hyperparameters.peak_lr, betas = hyperparameters.betas)



def format_info(info):
    tmp = ''
    for i, (key, value) in enumerate(info.items()):
        if i == 0:
            tmp += str(value).rjust(6) + ' ' + key
        else:
            tmp += '      ' + str(value).rjust(6) + ' ' + key
        
    return tmp



def train(settings, hyperparameters, model, dataset, device):
    print(colorama.Fore.GREEN)
    print('starting training')
    print(colorama.Style.RESET_ALL, end='')



    pin_device_args = {} if device == 'cpu' else {'pin_memory': True, 'pin_memory_device': device}

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size = settings.batch_size, shuffle = True, **pin_device_args)
    val_dataloader = torch.utils.data.DataLoader(dataset['validation'], batch_size = settings.batch_size, shuffle = True, **pin_device_args)


    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    if hasattr(dataset, 'ignore_index'):
        criterion.ignore_index = dataset.ignore_index

    
    optimizer = configure_optimizer(model, hyperparameters)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = hyperparameters.start_lr / hyperparameters.peak_lr, end_factor = 1.0, total_iters = hyperparameters.lr_warmup_steps // settings.schedule_every),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = hyperparameters.lr_decay_steps // settings.schedule_every, eta_min = hyperparameters.end_lr)
        ]
    )



    log_train_loss_sum = 0.0
    print_train_loss_sum = 0.0
    last_val_loss = 'n/a'


    start = time.time()
    for step in range(settings.total_steps):
        model.train()

        ids = next(iter(train_dataloader))['ids'].to(device)
        
        inputs, labels = ids[..., :-1], ids[..., 1:]

        logits, _ = model(inputs)

        # flatten batch and sequence dimensions into one dimension for computing the loss
        loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))


        log_train_loss_sum += loss.item()
        print_train_loss_sum += loss.item()


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (step + 1) % settings.update_every == 0:
            optimizer.step()
            optimizer.zero_grad()


        if (step + 1) % settings.schedule_every == 0:
            scheduler.step()

            log_msg = f'step: {step + 1}' + '  ' + 'lr: ' + '{:.6f}'.format(scheduler.get_last_lr()[0])
            with open(settings.train_log_path + '/stats/log.txt', 'a') as file: 
                file.write(log_msg + '\n')
        

        if (step + 1) % settings.log_loss_every == 0:
            train_loss_average = log_train_loss_sum / settings.log_loss_every
            log_train_loss_sum = 0.0

            log_msg = f'step: {step + 1}' + '  ' + 'train loss: ' + '{:.6f}'.format(train_loss_average)
            with open(settings.train_log_path + '/stats/log.txt', 'a') as file: 
                file.write(log_msg + '\n')


        if (step + 1) % settings.eval_every == 0:
            val_loss_sum = 0.0

            model.eval()
            with torch.inference_mode():
                for val_item in iter(val_dataloader):
                    val_ids = val_item['ids'].to(device)
                    val_inputs, val_labels = val_ids[..., :-1], val_ids[..., 1:]

                    val_logits, _ = model(val_inputs)

                    val_loss = criterion(val_logits.flatten(-3, -2), val_labels.flatten(-2, -1))

                    val_loss_sum += val_loss.item()

            model.train()
            
            val_loss_avg = val_loss_sum / len(val_dataloader)
            
            last_val_loss = val_loss_avg

            log_msg = f'step: {step + 1}' + '  ' + 'val loss: ' + '{:.6f}'.format(val_loss_avg)
            with open(settings.train_log_path + '/stats/log.txt', 'a') as file: 
                file.write(log_msg + '\n')

        
        if (step + 1) % settings.print_progress_every == 0:
            print_info = {
                'time elapsed': time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                'done': '{:.2f}'.format(100 * (step + 1) / settings.total_steps) + '%',
                'steps': step + 1,
                'epochs': (step + 1) // len(train_dataloader),
            }


            print_info['avg train loss'] = '{:.4f}'.format(print_train_loss_sum / settings.print_progress_every)
            print_train_loss_sum = 0

            print_info['last val loss'] = '{:.4f}'.format(last_val_loss) if last_val_loss != 'n/a' else 'n/a'

            print_info['current lr'] = '{:.6f}'.format(scheduler.get_last_lr()[0])

            # print the info
            print(colorama.Fore.YELLOW, end='')
            print(format_info(print_info))
            print(colorama.Style.RESET_ALL, end='')
        
        

        if (step + 1) % settings.save_checkpoint_every == 0:
            torch.save(model.state_dict(), os.path.join(settings.train_log_path, 'models', 'checkpoint-'  + 'step-' + str(step + 1) + '.pt'))
