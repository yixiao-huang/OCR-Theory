from einops import rearrange
import os 
from data_utils import ImplicitReasoningTask
import torch
import torch.nn as nn
from base_nonlinear import TF, DisentangledTF
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import glob
from typing import Optional
import matplotlib.pyplot as plt
import math
from typing import Optional
import numpy as np
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def criterion(f, y, mean = True, reduce = True):
    loss = torch.sum(y * -torch.log(f + 1e-8), dim = 1) 
    if not reduce:
        return loss
    if mean:
        return torch.mean(loss)
    return torch.sum(loss)

def relation_loss(f, y, mean = True, reduce = True, y_lo = 2, num_y = 4):
    rel_y = torch.zeros_like(f)
    rel_y[:, y_lo : y_lo + num_y] = 1
    loss = torch.mean(-torch.log(torch.sum(f * rel_y, dim = -1) + 1e-8))
    return loss
def subject_loss(name_animal_map, name_city_map, f, x, mean = True, reduce = True):
    name_map = {}
    for name, animal, city in zip(name_animal_map.keys(), name_animal_map.values(), name_city_map.values()):
        name_map[name] = [animal, city]
    subj_y = torch.zeros_like(f)
    for i in range(x.size(0)):
        # find the name in x
        name = None
        for j in range(x.size(1)):
            if x[i, j].item() in name_map.keys():
                name = x[i, j].item()
                break
        if name is None:
            raise ValueError("Name not found in the input")
        subj_y[i, name_map[name][0]] = 1
        subj_y[i, name_map[name][1]] = 1
    loss = torch.mean(-torch.log(torch.sum(f * subj_y, dim = -1) + 1e-8))
    return loss

def main(
    seed: int = 42,
    # task parameters
    num_name: int = 16,
    num_city: int = 4,
    num_animal: int = 4,
    num_probing: int = 0,
    num_noise: int = 4,
    num_signal: int = 2,
    seq_len: int = 16,
    # even_distribution: bool = False,
    use_eos: bool = True,
    train_test_split: float = 0.75,
    # model_parameters
    num_layers: int = 1,
    dim_d: int = 32,
    dim_r: Optional[int] = None,
    attn_heads: str = None,
    wte_type: str = 'ortho', # ['ortho', 'random']
    weight_tying: bool = True,
    use_pos: bool = False, # use positional encoding
    use_mlp: bool = True,
    mlp_width: int = 64,
    weight_init: str = 'zero', # ['zero', 'default']
    reparameterize_qk: bool = False, # combine value and output matrices
    reparameterize_ov: bool = False, # combine value and output matrices
    attn_arc: str = 'softmax', # ['softmax', 'linear']
    residual: bool = False, # residual connection
    disentangle: bool = False, # concat the residual connection
    layer_norm: bool = False, # layer norm
    # trainig parameters
    optim: str = 'adam',
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    scheduler: str = 'constant', # ['constant', 'cosine']
    num_steps: int = 10000,
    n_save: int = 10,
    log_every: int = 100,
    eval_size: int = 100,
    train_mode: str = 'fact_first', # ['fact_first', 'impl_first', 'mixed', 'mixed_finetune]
    early_stopping: bool = False,
    alt_ft_cutoff: float = 0.2, 
    # fine tuning parameters
    fine_tune: bool = False,
    finetune_lr: Optional[float] = None,
    finetune_weight_decay: Optional[float] = None,
    finetune_num_steps: Optional[int] = None,
    finetune_batch_size: Optional[int] = None,
    finetune_scheduler: Optional[str] = None,

    load_step: Optional[int] = None,
    # probing
    probing: bool = False,

    save_path: str = 'model_logs/',
    device: str = 'cuda',
    debug: bool = False,
    # log loss
    log_loss: bool = False,
):
    set_seed(seed)
    if attn_heads is None:
        attn_heads = [1] * num_layers
    else:
        attn_heads = list(map(int, attn_heads.split(',')))
    
    train_wandb_formatter = '{}_{}signal_{}name_{}city_{}animal_{}noise_{}eos_{}layers_{}mlp_{}heads_{}d_{}repq_{}repo_{}arc_{}resid_{}lr_{}wd_{}batch_{}seq_{}steps_{}init_{}{}'

    if fine_tune or probing:
        finetune_lr = finetune_lr if finetune_lr is not None else lr
        finetune_weight_decay = finetune_weight_decay if finetune_weight_decay is not None else weight_decay
        finetune_num_steps = finetune_num_steps if finetune_num_steps is not None else num_steps
        finetune_batch_size = finetune_batch_size if finetune_batch_size is not None else batch_size
        finetune_scheduler = finetune_scheduler if finetune_scheduler is not None else scheduler
        wandb_name_formatter = '{}_' + train_wandb_formatter
        wandb_name = wandb_name_formatter.format(
            'finetune' if fine_tune else 'probing',
            train_mode, 
            num_signal, num_name, num_city, num_animal, num_noise, 'no' if not use_eos else 'yes',
            num_layers, "no" if not use_mlp else mlp_width,
            ','.join(map(str, attn_heads)), dim_d, 
            'no' if not reparameterize_qk else 'yes', 'no' if not reparameterize_ov else 'yes',
            attn_arc, residual,
            finetune_lr, finetune_weight_decay, finetune_batch_size, seq_len, finetune_num_steps, 
            weight_init,
            optim,
            "_debug" if debug else ""
        )
        train_wandb_name = train_wandb_formatter.format(
            train_mode, 
            num_signal, num_name, num_city, num_animal, num_noise, 'no' if not use_eos else 'yes',
            num_layers, "no" if not use_mlp else mlp_width, ','.join(map(str, attn_heads)), dim_d, 
            'no' if not reparameterize_qk else 'yes', 'no' if not reparameterize_ov else 'yes', attn_arc, residual,
            lr, weight_decay, batch_size, seq_len, num_steps, weight_init, optim,
            "_debug" if debug else ""
        )
        if dim_r is not None:
            wandb_name = wandb_name.replace('{}d'.format(dim_d), '{}d_{}r'.format(dim_d, dim_r))
            train_wandb_name = train_wandb_name.replace('{}d'.format(dim_d), '{}d_{}r'.format(dim_d, dim_r))
        if layer_norm:
            wandb_name = wandb_name.replace('{}resid'.format(residual), '{}resid_ln'.format(residual))
            train_wandb_name = train_wandb_name.replace('{}resid'.format(residual), '{}resid_ln'.format(residual))
        if disentangle:
            wandb_name = wandb_name.replace('{}resid'.format(residual), '{}resid_disentangle'.format(residual))
            train_wandb_name = train_wandb_name.replace('{}resid'.format(residual), '{}resid_disentangle'.format(residual))
        if train_test_split != 0.75:
            wandb_name = wandb_name.replace('{}seq'.format(seq_len), '{}seq_{}split'.format(seq_len, train_test_split))
            train_wandb_name = train_wandb_name.replace('{}seq'.format(seq_len), '{}seq_{}split'.format(seq_len, train_test_split))
    else:
        wandb_name = train_wandb_formatter.format(
            train_mode, 
            num_signal, num_name, num_city, num_animal, num_noise, 'no' if not use_eos else 'yes',
            num_layers, "no" if not use_mlp else mlp_width, ','.join(map(str, attn_heads)), dim_d, 
            'no' if not reparameterize_qk else 'yes', 'no' if not reparameterize_ov else 'yes', attn_arc, residual,
            lr, weight_decay, batch_size, seq_len, num_steps, weight_init, optim,
            "_debug" if debug else ""
        )
        if dim_r is not None:
            wandb_name = wandb_name.replace('{}d'.format(dim_d), '{}d_{}r'.format(dim_d, dim_r))
        if layer_norm:
            wandb_name = wandb_name.replace('{}resid'.format(residual), '{}resid_ln'.format(residual))
        if disentangle:
            wandb_name = wandb_name.replace('{}resid'.format(residual), '{}resid_disentangle'.format(residual))
        if train_test_split != 0.75:
            wandb_name = wandb_name.replace('{}seq'.format(seq_len), '{}seq_{}split'.format(seq_len, train_test_split))
    
    wandb.init(project="implicit-reasoning", config=locals(), name = wandb_name)

    task = ImplicitReasoningTask(num_name, num_city, num_animal,
                        num_signal = num_signal,
                        num_probing = num_probing,
                        num_noise = num_noise,
                        seed = seed,
                        # even_distribution = even_distribution,
                        use_eos = use_eos,
                        train_test_split = train_test_split
            )
    if disentangle:
        model = DisentangledTF(seq_len = seq_len, dim_d = dim_d, vocab_size = task.vocab_size, 
                dim_r = dim_r,
                n_layer = num_layers, heads = attn_heads, 
                wte_type = wte_type, weight_tying = weight_tying, use_pos = use_pos,
                use_mlp = use_mlp, mlp_width = mlp_width,
                init=weight_init,
                reparameterize_qk = reparameterize_qk,
                reparameterize_ov = reparameterize_ov,
                attn_arc = attn_arc,
                residual = residual,
                layer_norm=layer_norm,
                device = device).cuda()
    else:
        model = TF(seq_len = seq_len, dim_d = dim_d, vocab_size = task.vocab_size, 
                    dim_r = dim_r,
                    n_layer = num_layers, heads = attn_heads, 
                    wte_type = wte_type, weight_tying = weight_tying, use_pos = use_pos,
                    use_mlp = use_mlp, mlp_width = mlp_width,
                    init=weight_init,
                    reparameterize_qk = reparameterize_qk,
                    reparameterize_ov = reparameterize_ov,
                    attn_arc = attn_arc,
                    residual = residual,
                    layer_norm=layer_norm,
                    device = device).cuda()

    if fine_tune or probing:
        def get_latest_epoch(loadpath):
            states = glob.glob1(loadpath, 'model_*')
            latest_epoch = -1
            for state in states:
                epoch = int(state.replace('model_', '').replace('.pth', ''))
                latest_epoch = max(epoch, latest_epoch)
            return latest_epoch
        if load_step is None:
            if os.path.exists(os.path.join(save_path, train_wandb_name, f'model_final.pth')):
                load_step = 'final'
            else:
                load_step = get_latest_epoch(os.path.join(save_path, train_wandb_name))
            
        print("Loading model from {}".format(os.path.join(save_path, train_wandb_name, f'model_{load_step}.pth')))
        model.load_state_dict(torch.load(os.path.join(save_path, train_wandb_name, f'model_{load_step}.pth')))
        
        # replace the training parameters with fine tuning parameters
        lr = finetune_lr
        weight_decay = finetune_weight_decay
        num_steps = finetune_num_steps
        batch_size = finetune_batch_size
        scheduler = finetune_scheduler

    save_path = os.path.join(save_path, wandb_name)
    os.makedirs(save_path, exist_ok = True)

    def freeze_model(layer = 0, freeze = True, component = 'attn'):
        if component == 'attn':
            model.self_attn[layer].w_q.weight.requires_grad = not freeze
            if not reparameterize_qk:
                model.self_attn[layer].w_k.weight.requires_grad = not freeze
            # model.self_attn[layer].w_v.weight.requires_grad = not freeze
            model.self_attn[layer].w_o.weight.requires_grad = not freeze
            if not reparameterize_ov:
                # model.self_attn[layer].w_o.weight.requires_grad = not freeze
                model.self_attn[layer].w_v.weight.requires_grad = not freeze
        elif component == 'mlp':
            model.mlp[layer].layer1.weight.requires_grad = not freeze
            model.mlp[layer].layer2.weight.requires_grad = not freeze
            model.mlp[layer].layer1.bias.requires_grad = not freeze
        else:
            raise ValueError("Invalid component {}".format(component))
        
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    
    if 'mixed_alt_ft' in train_mode:
    # if train_mode == 'mixed_alt_ft':
        # properly initialize the model
        if num_layers == 2:
            for nl in range(num_layers):
                dh = model.self_attn[nl].w_q.weight.data.shape[0]
                model.self_attn[nl].w_q.weight.data = torch.eye(dh).cuda()
                if not reparameterize_qk:
                    model.self_attn[nl].w_k.weight.data = torch.eye(dh).cuda()
                model.self_attn[nl].w_v.weight.data = torch.eye(dh).cuda()
                if disentangle:
                    model.self_attn[nl].w_o.weight.data = torch.eye(dim_d).repeat(1, 2**(nl)).cuda()
                else:
                    model.self_attn[nl].w_o.weight.data = torch.eye(dh).cuda()
        if use_mlp:
            dh = model.mlp[0].layer1.weight.data.shape[0]
            model.mlp[0].layer1.weight.data = torch.eye(dh).cuda()
            model.mlp[0].layer2.weight.data = torch.eye(dh).cuda()
            if model.mlp[0].layer1.bias is not None:
                model.mlp[0].layer1.bias.data = torch.zeros(dh).cuda()
        if not 'nofreeze' in train_mode:
            freeze_model(0, freeze = False)
            if num_layers == 2:
                freeze_model(1, freeze = True)
            else:
                freeze_model(0, freeze = True, component = 'mlp')
    if optim == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if scheduler == 'cosine':
        lr_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    else:
        lr_scheduler = None
    save_every = num_steps // n_save
    if log_loss:
        loss_list = []

    for step in tqdm(range(num_steps), desc='Training'):
        model.train()
        if probing:
            X, y = task.get_batch(batch_size, seq_len, get_train = True, task = 'probing') 
        elif fine_tune:
            # only finetune on the fact part and evaluate on the implication part
            X, y = task.get_batch(batch_size, seq_len, get_train = False, task = 'fact')
        else:
            if (train_mode == 'fact_first' and step < num_steps // 2) or (train_mode == 'impl_first' and step >= num_steps // 2):
                X, y = task.get_batch(batch_size, seq_len, get_train = True, task = 'fact')
            elif (train_mode == 'fact_first' and step >= num_steps // 2) or (train_mode == 'impl_first' and step < num_steps // 2):
                X, y = task.get_batch(batch_size, seq_len, get_train = True, task = 'impl')
            elif 'mixed' in train_mode:
                # if train_mode == 'mixed' or 'mixed_finetune':
                if train_mode == 'mixed':
                    X_1, y_1 = task.get_batch(batch_size // 2, seq_len, get_train = True, task = 'fact')
                    X_2, y_2 = task.get_batch(batch_size // 2, seq_len, get_train = True, task = 'impl')
                    X_1, y_1 = X_1.to(device), y_1.to(device)
                    X_2, y_2 = X_2.to(device), y_2.to(device)
                    X = rearrange([X_1, X_2], 's b t -> (s b) t')
                    y = rearrange([y_1, y_2], 's b -> (s b)')
                elif train_mode == 'mixed_finetune':
                    ratio = 1 / 3.0
                    X_1, y_1 = task.get_batch(int(batch_size * ratio), seq_len, get_train = True, task = 'fact')
                    X_2, y_2 = task.get_batch(int(batch_size * ratio), seq_len, get_train = True, task = 'impl')
                    X_3, y_3 = task.get_batch(int(batch_size * (1 - 2 * ratio)), seq_len, get_train = False, task = 'fact')
                    X_1, y_1 = X_1.to(device), y_1.to(device)
                    X_2, y_2 = X_2.to(device), y_2.to(device)
                    X_3, y_3 = X_3.to(device), y_3.to(device)
                    assert len(X_1) + len(X_2) + len(X_3) == batch_size
                    X = torch.cat([X_1, X_2, X_3], dim=0)
                    y = torch.cat([y_1, y_2, y_3], dim=0)
                elif 'mixed_alt_ft' in train_mode:
                    cutoff = int(num_steps * alt_ft_cutoff)

                    if step < cutoff:
                        if step == 0:
                            if scheduler == 'cosine':
                                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cutoff)
                            else:
                                lr_scheduler = None
                        ratio = train_test_split / ( 1 + train_test_split)
                        if 'mixed_alt_ft_impl' in train_mode:
                            X, y = task.get_batch(batch_size, seq_len, get_train = True, task = 'impl')
                            X, y = X.to(device), y.to(device)
                        else:
                            X_1, y_1 = task.get_batch(int(batch_size * 0.5), seq_len, get_train = True, task = 'fact')
                            X_2, y_2 = task.get_batch(int(batch_size * 0.5), seq_len, get_train = False, task = 'fact')
                                # X_3, y_3 = task.get_batch(int(batch_size * (1 - 2 * ratio)), seq_len, get_train = False, task = 'fact')
                            X_1, y_1 = X_1.to(device), y_1.to(device)
                            X_2, y_2 = X_2.to(device), y_2.to(device)
                            X = torch.cat([X_1, X_2], dim=0)
                            y = torch.cat([y_1, y_2], dim=0)
                    else:
                        if step == cutoff:
                            # freeze the first layer and unfreeze the second layer
                            if not 'nofreeze' in train_mode:
                                freeze_model(0, freeze = True)
                                if num_layers == 2:
                                    freeze_model(1, freeze = False)
                                else:
                                    freeze_model(0, freeze = False, component = 'mlp')
                            if optim == 'adam':
                                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                            elif optim == 'sgd':
                                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                            if scheduler == 'cosine':
                                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps - cutoff)
                            else:
                                lr_scheduler = None
                        if num_probing > 0:
                            ratio = 1 / 6
                            X_1, y_1 = task.get_batch(int(batch_size * 2 * ratio), seq_len, get_train = True, task = 'fact')
                            X_2, y_2 = task.get_batch(int(batch_size * 2 * ratio), seq_len, get_train = True, task = 'impl')
                            X_3, y_3 = task.get_batch(int(batch_size * ratio), seq_len, get_train = True, task = 'probing')
                            X_4, y_4 = task.get_batch(int(batch_size * ratio), seq_len, get_train = False, task = 'fact')
                            X_1, y_1 = X_1.to(device), y_1.to(device)
                            X_2, y_2 = X_2.to(device), y_2.to(device)
                            X_3, y_3 = X_3.to(device), y_3.to(device)
                            X_4, y_4 = X_4.to(device), y_4.to(device)
                            X = torch.cat([X_1, X_2, X_3, X_4], dim=0)
                            y = torch.cat([y_1, y_2, y_3, y_4], dim=0)
                        else:
                            ratio = 1/3.0
                            X_1, y_1 = task.get_batch(int(batch_size * ratio), seq_len, get_train = True, task = 'fact')
                            X_2, y_2 = task.get_batch(int(batch_size * ratio), seq_len, get_train = True, task = 'impl')
                            X_3, y_3 = task.get_batch(int(batch_size * (1 - 2 * ratio)), seq_len, get_train = False, task = 'fact')
                            X_1, y_1 = X_1.to(device), y_1.to(device)
                            X_2, y_2 = X_2.to(device), y_2.to(device)
                            X_3, y_3 = X_3.to(device), y_3.to(device)
                            X = torch.cat([X_1, X_2, X_3], dim=0)
                            y = torch.cat([y_1, y_2, y_3], dim=0)
                else:
                    raise ValueError("Invalid train mode {}".format(train_mode))
                # perturb the order
                idx = torch.randperm(X.size(0))
                X, y = X[idx], y[idx]
            else:
                raise ValueError("Invalid train mode {}".format(train_mode))
        X, y = X.to(device), y.to(device)
        if step == 0:
            print("sample X: ", X[:5])
            print("sample y: ", y[:5])
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, F.one_hot(y, num_classes = out.shape[-1]).float())
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        if step % log_every == 0:
            model.eval()
            val_X_fact, val_y_fact = task.get_batch(eval_size, seq_len, get_train = True, task = 'fact')
            val_X_impl, val_y_impl = task.get_batch(eval_size, seq_len, get_train = True, task = 'impl')
            val_X_fact, val_y_fact = val_X_fact.to(device), val_y_fact.to(device)
            val_X_impl, val_y_impl = val_X_impl.to(device), val_y_impl.to(device)

            with torch.no_grad():
                val_out_fact = model(val_X_fact)
                val_out_impl = model(val_X_impl)

            val_loss_fact = criterion(val_out_fact, F.one_hot(val_y_fact, num_classes = out.shape[-1]).float())
            val_loss_impl = criterion(val_out_impl, F.one_hot(val_y_impl, num_classes = out.shape[-1]).float())
            val_loss_rel1 = relation_loss(val_out_fact, val_y_fact, y_lo = num_signal, num_y = num_city)
            val_loss_rel2 = relation_loss(val_out_impl, val_y_impl, y_lo = num_signal + num_city, num_y = num_animal)
            val_loss_rel = 0.5 * (val_loss_rel1 + val_loss_rel2)
            val_loss_subj = 0.5 * (subject_loss(task.name_animal_map, task.name_city_map, val_out_fact, val_X_fact) + subject_loss(task.name_animal_map, task.name_city_map, val_out_impl, val_X_impl))
            
            wandb_dict = {'loss/train_loss': loss.item(), 
                    'loss/val_loss': 0.5 * (val_loss_fact.item() + val_loss_impl.item()),
                    'loss/val_loss_fact': val_loss_fact.item(),
                    'loss/val_loss_impl': val_loss_impl.item(),
                    'seq_learning/val_loss_rel': val_loss_rel.item(),
                    'seq_learning/val_loss_subj': val_loss_subj.item(),
                    # 'val_loss': val_loss.item(), 
                    'lr': optimizer.param_groups[0]['lr'], 'step': step
                    }
            if num_probing > 0:
                val_X_probing, val_y_probing = task.get_batch(eval_size, seq_len, get_train = True, task = 'probing')
                val_X_probing, val_y_probing = val_X_probing.to(device), val_y_probing.to(device)
                with torch.no_grad():
                    val_out_probing = model(val_X_probing)
                val_loss_probing = criterion(val_out_probing, F.one_hot(val_y_probing, num_classes = out.shape[-1]).float())
                wandb_dict['loss/val_loss_probing'] = val_loss_probing.item()
            
            # weight norm
            for i in range(num_layers):
                attni = model.self_attn[i].w_q.weight.data.T.detach() @ model.self_attn[i].w_k.weight.data.detach()
                attni_norm = torch.norm(attni)
                plt.figure(figsize = (10, 10))
                plt.imshow(attni.cpu().numpy(), aspect = 'auto', interpolation = 'nearest')
                plt.colorbar()
                wandb_dict[f'weights/attn_{i+1}'] = wandb.Image(plt)
                plt.close()
                wandb_dict[f'norms/attn_{i+1}_norm'] = attni_norm.item()
                attn_mat = model.self_attn[i].w_v.weight.data.T.detach() @ model.self_attn[i].w_o.weight.data.T.detach()
                attn_mat_norm = torch.norm(attn_mat)
                wandb_dict[f'norms/v_{i+1}_norm'] = attn_mat_norm.item()

                # circuit 
                if i == 0:
                    w1_name_city = attn_mat[num_signal+num_city + num_animal + num_probing:num_signal+num_city + num_animal+ num_probing + num_name, num_signal:num_signal+num_city]
                    w1_name_animal = attn_mat[num_signal+num_city + num_animal + num_probing:num_signal+num_city + num_animal+ num_probing + num_name, num_signal+num_city:num_signal+num_city + num_animal]
                    w1_rel_pred = attn_mat[:num_signal, num_signal:num_signal+num_city + num_animal]
                    # norm: we expect the first to be larger than the second
                    w1_name_city_norm = torch.norm(w1_name_city)
                    w1_name_animal_norm = torch.norm(w1_name_animal)
                    w1_rel_pred_norm = torch.norm(w1_rel_pred)
                    wandb_dict[f'norms/w1_name_city_norm'] = w1_name_city_norm.item()
                    wandb_dict[f'norms/w1_name_animal_norm'] = w1_name_animal_norm.item()
                    wandb_dict[f'norms/w1_rel_pred_norm'] = w1_rel_pred_norm.item()
                    # visualization of the weights
                    fig, ax = plt.subplots(1, 2, figsize = (12, 6))
                    map1 = ax[0].imshow(w1_name_city.cpu().numpy())
                    # set subplot title
                    ax[0].set_title('w1_name_city')
                    # set colorbar
                    fig.colorbar(map1, ax=ax[0])
                    # plot second subplot
                    map2 = ax[1].imshow(w1_name_animal.cpu().numpy())
                    ax[1].set_title('w1_name_animal')
                    # set colorbar
                    fig.colorbar(map2, ax=ax[1])
                    wandb_dict[f'weights/w1_name_city_animal'] = wandb.Image(fig)
                    plt.close(fig)
                    fig, ax = plt.subplots(1, 1, figsize = (6, 12))
                    map3 = ax.imshow(w1_rel_pred.cpu().numpy(), aspect = 'auto')
                    ax.set_title('w1_rel_pred')
                    fig.colorbar(map3, ax=ax)
                    wandb_dict[f'weights/w1_rel_pred'] = wandb.Image(fig)
                    plt.close(fig)
                    plt.figure(figsize = (12, 12))
                    plt.imshow(attn_mat.cpu().numpy(), aspect = 'auto')
                    # Force x/y ticks to be integers
                    plt.xticks(ticks=np.arange(attn_mat.shape[1]))
                    plt.yticks(ticks=np.arange(attn_mat.shape[0]))
                    plt.colorbar()
                    plt.title('w1_raw')
                    wandb_dict[f'weights/w1_raw'] = wandb.Image(plt)
                    plt.close()
                    plt.figure(figsize = (10, 10))
                    plt.imshow(attni[-1:].cpu().numpy(), aspect = 'auto')
                    plt.colorbar()
                    plt.title('attn_mat')
                    wandb_dict[f'weights/attn1_eos2all'] = wandb.Image(plt)
                    plt.close()
                elif i == 1:
                    # We expect this to be as small as possible
                    w2_name_city_animal = attn_mat[num_signal+num_city + num_animal + num_probing:num_signal+num_city + num_animal+ num_probing + num_name, num_signal:num_signal+num_city + num_animal]
                    w2_name_city_animal_norm = torch.norm(w2_name_city_animal)
                    w2_pred_pred = attn_mat[:num_signal+num_city + num_animal, :num_signal+num_city + num_animal]
                    # w2_animal_pred = attn_mat[num_signal+num_city:num_signal+num_city + num_animal, num_signal: num_signal + num_city + num_animal]
                    # w2_rel_city_pred_norm = torch.norm(w2_rel_city_pred)
                    # w2_animal_pred_norm = torch.norm(w2_animal_pred)
                    w2_pred_pred_norm = torch.norm(w2_pred_pred)
                    wandb_dict[f'norms/w2_name_city_animal_norm'] = w2_name_city_animal_norm.item()
                    # wandb_dict[f'norms/w2_rel_city_pred_norm'] = w2_rel_city_pred_norm.item()
                    # wandb_dict[f'norms/w2_animal_pred_norm'] = w2_animal_pred_norm.item()
                    wandb_dict[f'norms/w2_pred_pred_norm'] = w2_pred_pred_norm.item()
                    # visualization of the weights
                    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
                    w2_map = ax.imshow(w2_name_city_animal.cpu().numpy())
                    ax.set_title('w2_name_city_animal')
                    fig.colorbar(w2_map, ax=ax)
                    wandb_dict[f'weights/w2_name_city_animal'] = wandb.Image(fig)
                    plt.close(fig)
                    # fig, ax = plt.subplots(1, 2, figsize = (12, 6))
                    # w2_rel_city_pred_map = ax[0].imshow(w2_rel_city_pred.cpu().numpy())
                    # ax[0].set_title('w2_rel_city_pred')
                    # fig.colorbar(w2_rel_city_pred_map, ax=ax[0])
                    # w2_animal_pred_map = ax[1].imshow(w2_animal_pred.cpu().numpy())
                    # ax[1].set_title('w2_animal_pred')
                    # fig.colorbar(w2_animal_pred_map, ax=ax[1])
                    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
                    w2_pred_pred_map = ax.imshow(w2_pred_pred.cpu().numpy())
                    ax.set_title('w2_pred_pred')
                    fig.colorbar(w2_pred_pred_map, ax=ax)
                    wandb_dict[f'weights/w2_pred_pred'] = wandb.Image(fig)
                    plt.close(fig)
                    plt.figure(figsize = (12, 12))
                    plt.imshow(attn_mat.cpu().numpy())
                    plt.colorbar()
                    plt.title('w2_raw')
                    wandb_dict[f'weights/w2_raw'] = wandb.Image(plt)
                    plt.close()

                    plt.figure(figsize = (10, 10))
                    plt.imshow(attni[num_signal : num_signal + num_city].cpu().numpy())
                    plt.colorbar()
                    plt.title('attn_mat')
                    wandb_dict[f'weights/attn2_city2all'] = wandb.Image(plt)
                    plt.close()
            if probing:
                val_X_probing, val_y_probing = task.get_batch(eval_size, seq_len, get_train = True, task = 'probing')
                test_X_probing, test_y_probing = task.get_batch(eval_size, seq_len, get_train = False, task = 'probing')
                val_X_probing, val_y_probing = val_X_probing.to(device), val_y_probing.to(device)
                test_X_probing, test_y_probing = test_X_probing.to(device), test_y_probing.to(device)
                with torch.no_grad():
                    val_out_probing = model(val_X_probing)
                    test_out_probing = model(test_X_probing)
                val_loss_probing = criterion(val_out_probing, F.one_hot(val_y_probing, num_classes = out.shape[-1]).float())
                test_loss_probing = criterion(test_out_probing, F.one_hot(test_y_probing, num_classes = out.shape[-1]).float())
                wandb_dict['val_loss_probing'] = val_loss_probing.item()
                wandb_dict['test_loss_probing'] = test_loss_probing.item()
            
            if fine_tune or train_mode == 'mixed_finetune' or 'mixed_alt_ft' in train_mode:
                test_X_fact, test_y_fact = task.get_batch(eval_size, seq_len, get_train = False, task = 'fact')
                test_X_impl, test_y_impl = task.get_batch(eval_size, seq_len, get_train = False, task = 'impl')
                test_X_fact, test_y_fact = test_X_fact.to(device), test_y_fact.to(device)
                test_X_impl, test_y_impl = test_X_impl.to(device), test_y_impl.to(device)
                with torch.no_grad():
                    test_out_fact = model(test_X_fact)
                    test_out_impl = model(test_X_impl)
                test_loss_fact = criterion(test_out_fact, F.one_hot(test_y_fact, num_classes = out.shape[-1]).float())
                test_loss_impl = criterion(test_out_impl, F.one_hot(test_y_impl, num_classes = out.shape[-1]).float())
                wandb_dict['test_loss_fact'] = test_loss_fact.item()
                wandb_dict['test_loss_impl'] = test_loss_impl.item()
                if num_probing > 0:
                    test_X_probing, test_y_probing = task.get_batch(eval_size, seq_len, get_train = False, task = 'probing')
                    test_X_probing, test_y_probing = test_X_probing.to(device), test_y_probing.to(device)
                    with torch.no_grad():
                        test_out_probing = model(test_X_probing)
                    test_loss_probing = criterion(test_out_probing, F.one_hot(test_y_probing, num_classes = out.shape[-1]).float())
                    wandb_dict['test_loss_probing'] = test_loss_probing.item()
                test_loss_rel = relation_loss(test_out_impl, test_y_impl, y_lo = num_signal + num_city, num_y = num_animal)
                test_loss_subj = subject_loss(task.name_animal_map, task.name_city_map, test_out_impl, test_X_impl)
                wandb_dict['test_loss_rel'] = test_loss_rel.item()
                wandb_dict['test_loss_subj'] = test_loss_subj.item()
            wandb.log(wandb_dict, step = step)
            if log_loss:
                loss_list.append(
                    {
                        "test_loss_fact": test_loss_fact.item() if fine_tune or train_mode == 'mixed_finetune' or 'mixed_alt_ft' in train_mode else None,
                        "test_loss_impl": test_loss_impl.item() if fine_tune or train_mode == 'mixed_finetune' or 'mixed_alt_ft' in train_mode else None,
                        "val_loss_fact": val_loss_fact.item(),
                        "val_loss_impl": val_loss_impl.item(),
                    }
                )
            if fine_tune:
                print(" \
                    Test Fact Loss: {:.4f} \
                    Test Impl Loss: {:.4f} \
                    ".format(test_loss_fact.item(), test_loss_impl.item()))
            if probing:
                print(" \
                    Val Probing Loss: {:.4f} \
                    Test Probing Loss: {:.4f} \
                    ".format(val_loss_probing.item(), test_loss_probing.item()))
            print(" \
                Step: {} \
                Train Loss: {:.4f} \
                Val (Fact + Impl) Loss: {:.4f} \
                Fact Val Loss: {:.4f} \
                Impl Val Loss: {:.4f} \
                Subj Val Loss: {:.4f} \
                Rel Val Loss: {:.4f} \
                LR: {:.5f} \
                ".format(step, loss.item(), 
                    0.5 * (val_loss_fact.item() + val_loss_impl.item()),
                    val_loss_fact.item(), val_loss_impl.item(), 
                    val_loss_subj.item(), val_loss_rel.item(),
                    optimizer.param_groups[0]['lr']))
            # print(" \
            #     Val Loss: {:.4f} \
            #     Subj + Rel Val Loss: {:.4f} \
            #     ".format((val_loss_fact.item() + val_loss_impl.item())*0.5, val_loss_subj.item() + val_loss_rel.item()))

        if step % save_every == 0:
            torch.save(model.state_dict(), f"{save_path}/model_{step}.pth")
        if log_loss:
            with open(f"{save_path}/loss.json", 'w') as f:
                json.dump(loss_list, f, indent=4)
        # early stopping
        if early_stopping and step > num_steps * 0.2:
            if fine_tune: 
                if test_out_fact < 1e-4:
                    print("Early stopping at step {}".format(step))
                    break
            elif probing:
                pass 
                # if val_loss_probing < 1e-4:
                #     print("Early stopping at step {}".format(step))
                #     break
            elif val_loss_fact < 1e-4 and val_loss_impl < 1e-4:
                print("Early stopping at step {}".format(step))
                break
    torch.save(model.state_dict(), f"{save_path}/model_final.pth")
    if log_loss:
        with open(f"{save_path}/loss.json", 'w') as f:
            json.dump(loss_list, f, indent=4)
        # with open(f"paper_loss/loss_{dim_r}.json", 'w') as f:
        #     json.dump(loss_list, f, indent=4)
    # inspecting model weights

import tyro
tyro.cli(main)