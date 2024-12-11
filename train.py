import torch
import numpy as np

import time
import colorama


from dataclasses import dataclass
from matplotlib import pyplot
from typing import Optional

from sample import sample


@dataclass
class train_config:
    sequence_length: int
    batch_size: int


    total_steps: int = 5000


    update_every: int = 1

    schedule_every: int = 10


    log_minor: bool = True
    log_major: bool = True

    log_minor_every: int = 20
    log_major_every: int = 200

    sample_at_log_minor: bool = False
    save_stats_to_file_at_log_minor: bool = False

    sample_at_log_major: bool = False
    plot_at_log_major: bool = True
    save_at_log_major: bool = False
    save_stats_to_file_at_log_major: bool = True


    train_run_path: Optional[str] = None
    
    


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
        # {
        #     'params': get_params_with_names(model.named_parameters(), ['state_matrices_up']),
        #     'lr': hyperparameters.peak_lr / 5
        # },
        {
            'params': remove_params_with_names(model.named_parameters(), nodecay_param_names),# + ['state_matrices_up']),
            'weight_decay': hyperparameters.weight_decay
        },
        {
            'params': get_params_with_names(model.named_parameters(), nodecay_param_names),
            'weight_decay': 0.0
        }
    ]

    return torch.optim.AdamW(optim_groups, lr = hyperparameters.peak_lr, betas = hyperparameters.betas)



def format_info(info):
    str = ''
    for i, (key, value) in enumerate(info.items()):
        if i == 0:
            str += value + ' ' + key
        else:
            str += (value + ' ' + key).rjust(len(key) + 12)
        
    return str



def train(settings, hyperparameters, model, dataset, tokenizer, device):
    print(colorama.Fore.GREEN)
    print('starting training')
    print(colorama.Style.RESET_ALL, end='')


    start = time.time()

    pin_device_args = {}
    if device != 'cpu':
        pin_device_args['pin_memory'] = True
        pin_device_args['pin_memory_device'] = device

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size = settings.batch_size, shuffle = True, **pin_device_args)
    val_dataloader = torch.utils.data.DataLoader(dataset['validation'], batch_size = settings.batch_size, shuffle = True, **pin_device_args)


    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    if tokenizer.padding:
        criterion.ignore_index = tokenizer.token_to_id('<pad>')
    

    
    optimizer = configure_optimizer(model, hyperparameters)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = hyperparameters.start_lr / hyperparameters.peak_lr, end_factor = 1.0, total_iters = hyperparameters.lr_warmup_steps // settings.schedule_every),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = hyperparameters.lr_decay_steps // settings.schedule_every, eta_min = hyperparameters.end_lr)
        ]
    )

    
    train_loss_history = torch.empty(settings.total_steps // settings.log_major_every, device='cpu')
    val_loss_history = torch.empty(settings.total_steps // settings.log_major_every, device='cpu')
    
    log_major_loss_total = 0.0
    log_minor_loss_total = 0.0

    
    for step in range(settings.total_steps):
        model.train()

        inputs, labels = next(iter(train_dataloader)).values()
        
        inputs, labels = inputs.to(device), labels.to(device)

        logits, _ = model(inputs, model.get_initial_state())

        # flatten batch and sequence dimensions into one dimension for computing the loss
        loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))


        # log the loss
        log_major_loss_total += loss.item()
        log_minor_loss_total += loss.item()


        loss.backward()

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (step + 1) % settings.update_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % settings.schedule_every == 0:
            scheduler.step()


        with torch.inference_mode():
            model.eval()

            # log information
            if settings.log_minor and (step + 1) % settings.log_minor_every == 0:
                # compute the average loss
                avg_train_loss = log_minor_loss_total / settings.log_minor_every
                log_minor_loss_total = 0.0
                

                # set the info to be displayed
                info = {
                    'time elapsed': time.strftime('%M:%S', time.gmtime(time.time() - start)),
                    'steps': str(step + 1),
                    'done': '{:.2f}'.format(100 * (step + 1) / settings.total_steps) + '%',
                    'avg train loss': '{:.4f}'.format(avg_train_loss),
                    'lr': '{:.4f}'.format(scheduler.get_last_lr()[0])
                }


                # print the info
                print(colorama.Fore.CYAN, end='')
                print(format_info(info))
                print(colorama.Style.RESET_ALL, end='')


                # print a sample from the model
                if settings.sample_at_log_minor:
                    print()
                    print(colorama.Fore.GREEN, end='')
                    print('sample:')
                    print(colorama.Style.RESET_ALL, end='')
                    sequence_start = None
                    if hasattr(tokenizer, 'character_level'):
                        sequence_start = 'A'
                    else:
                        sequence_start = '<bos>'
                    print(sample(model, tokenizer, sequence_start = sequence_start, temperature = 1.0, max_length = settings.sequence_length, device = device))
                    print()
                

                # save the info to a file
                if settings.save_stats_to_file_at_log_minor:
                    with open(settings.train_run_path + '/stats/minor-log.txt', 'a') as f:
                        f.write(format_info(info) + '\n')



            if settings.log_major and (step + 1) % settings.log_major_every == 0:
                # compute the average loss
                avg_train_loss = log_major_loss_total / settings.log_major_every
                log_major_loss_total = 0.0


                # evaluate the model on the validation set
                val_loss_total = 0.0
                for val_step in range(len(val_dataloader)):
                    inputs, labels = next(iter(val_dataloader)).values()
                
                    inputs, labels = inputs.to(device), labels.to(device)

                    logits, _ = model(inputs, model.get_initial_state())

                    loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))

                    val_loss_total += loss.item()
                avg_val_loss = val_loss_total / len(val_dataloader)


                # set the info to be displayed
                info = {
                    'time elapsed': time.strftime('%M:%S', time.gmtime(time.time() - start)),
                    'steps': str(step + 1),
                    'done': '{:.2f}'.format(100 * (step + 1) / settings.total_steps) + '%',
                    'avg train loss': '{:.4f}'.format(avg_train_loss),
                    'avg val loss': '{:.4f}'.format(avg_val_loss),
                    'epochs': str((step + 1) // len(train_dataloader)),
                    'lr': '{:.4f}'.format(scheduler.get_last_lr()[0])
                }


                # print the info
                print(colorama.Fore.YELLOW, end='')
                print(format_info(info))
                print(colorama.Style.RESET_ALL, end='')


                # set the loss history
                train_loss_history[step // settings.log_major_every] = avg_train_loss
                val_loss_history[step // settings.log_major_every] = avg_val_loss

                # save the model
                if settings.save_at_log_major:
                    torch.save(model.state_dict(), settings.train_run_path + '/models/' + 'step-' + str(step + 1) + '.pt')


                # plot the loss
                if settings.plot_at_log_major and step > settings.log_major_every:
                    x_axis_steps = np.arange(settings.log_major_every, step + 1 + settings.log_major_every, settings.log_major_every)

                    pyplot.plot(x_axis_steps, train_loss_history[:((step + 1) // settings.log_major_every)].numpy(), label='train', color='blue')
                    pyplot.plot(x_axis_steps, val_loss_history[:((step + 1) // settings.log_major_every)].numpy(), label='val', color='red')

                    # set the plot scale to log
                    # pyplot.yscale('log')
                    
                    pyplot.title('Loss vs. Step')
                    pyplot.xlabel('Step')
                    pyplot.ylabel('Loss')

                    # set the legend
                    pyplot.legend()
                    
                    # save the plot
                    pyplot.savefig(settings.train_run_path + '/stats/loss plot.png')

                    pyplot.clf()

                model.train()

                

                # print a sample from the model
                if settings.sample_at_log_major:
                    print()
                    print(colorama.Fore.GREEN, end='')
                    print('sample:')
                    print(colorama.Style.RESET_ALL, end='')
                    sequence_start = None
                    if hasattr(tokenizer, 'character_level'):
                        sequence_start = 'A'
                    else:
                        sequence_start = '<bos>'
                    print(sample(model, tokenizer, sequence_start = sequence_start, temperature = 1.0, max_length = settings.sequence_length, device = device))
                    print()

                # save the info to a file
                if settings.save_stats_to_file_at_log_major:
                    with open(settings.train_run_path + '/stats/major-log.txt', 'a') as f:
                        f.write(format_info(info) + '\n')



