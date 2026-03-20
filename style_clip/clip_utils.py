#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import numpy as np
import time
import datetime
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.distributed as dist

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from style_clip.distributed import is_dist_avail_and_initialized


# https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def collect_params(model):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Embedding)
    blacklist_weight_modules = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                                torch.nn.SyncBatchNorm, torch.nn.LayerNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif 'class_embedding' in pn:
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)
    assert len(param_dict) == len(union_params)

    # double check for p.ndim < 2 from openclip
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    for n, p in model.named_parameters():
        if exclude(n, p):
            assert n in no_decay

    print("weight params:\n{}".format('\n'.join(sorted(decay))))
    print("bn and bias params without decay:\n{}".format('\n'.join(sorted(no_decay))))

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad], "weight_decay": 0.0},
    ]

    return optim_groups


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def split_reshape(x, bs, combination=None):
    n = len(x) // bs
    assert n in [2, 3], "The num augs should be 2 or 3 in number"
    f = torch.split(x, [bs] * n, dim=0)
    if combination is None:
        x_reshape = torch.cat([f[i].unsqueeze(1) for i in range(n)], dim=1)
    else:
        x_reshape = torch.cat([f[i].unsqueeze(1) for i in combination], dim=1)

    # if repeatcase:
    #     x_reshape = torch.cat([f1.unsqueeze(1), f1.unsqueeze(1)], dim=1)
    return x_reshape


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# Copy from https://github.com/learn2phoenix/dynamicDistances/blob/main/metrics/metrics.py
class Metrics(object):
    def __init__(self):
        self.data = None

    @staticmethod
    def get_recall(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        found = np.where(np.amin(np.absolute(preds), axis=1) == 0)[0]
        return found.shape[0] / gts.shape[0]

    @staticmethod
    def get_mrr(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1 / valid_cols)

    @staticmethod
    def get_map(preds, gts, topk=5):
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]

    @staticmethod
    def get_recall_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        found = np.where(np.amax(preds, axis=1) == True)[0]
        return found.shape[0] / preds.shape[0]

    @staticmethod
    def get_mrr_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1 / valid_cols)

    @staticmethod
    def get_map_bin(preds, topk=5):
        # preds is a binary matrix of size Q x K
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]


def load_network(model, path, checkpoint_key="net", strict=True):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu")

        # rename pre-trained keys
        state_dict = checkpoint[checkpoint_key]
        state_dict_new = {k.replace("module.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict_new, strict=strict)
        assert msg.missing_keys == []

        print("=> loaded weights model '{}'".format(path))
    else:
        print("=> no weights found at '{}'".format(path))


if __name__ == '__main__':
    pass
