#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-14
"""

import copy
from typing import Callable

import torch
from torch import autograd
from torch import nn


class MAML(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 lr=0.1,
                 num_steps=3):
        super(MAML, self).__init__()
        self.model = model
        self._loss_fn = loss_fn
        self._lr = lr
        self._num_steps = num_steps

        memo = {id(p): p for p in model.parameters()}
        self._model_symbol = copy.deepcopy(model, memo)

        self._param_spec = {}
        self._make_param_spec(self._model_symbol)
        self._param_list = list(self._param_spec.keys())

    def _make_param_spec(self, module):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, nn.Parameter):
                delattr(module, name)
                self._param_spec[obj] = (module, name)
        for child in module.children():
            self._make_param_spec(child)

    def forward(self, support_x, support_y, query_x, query_y):
        return torch.mean(torch.stack([
            self._per_task(sx, sy, qx, qy)
            for sx, sy, qx, qy in zip(support_x, support_y, query_x, query_y)
        ]))

    def _per_task(self, support_x, support_y, query_x, query_y):
        for i in range(self._num_steps):
            pred_y = self.model(support_x) if i == 0 else self._model_symbol(support_x)
            loss = self._loss_fn(pred_y, support_y)
            if len(loss.shape) != 0:
                loss = loss.mean()

            grad_list = autograd.grad(loss, self._param_list)
            for param, grad in zip(self._param_list, grad_list):
                new_param = param - self._lr * grad
                module, name = self._param_spec[param]
                setattr(module, name, new_param)

        pred_y = self._model_symbol(query_x)
        loss = self._loss_fn(pred_y, query_y)
        if len(loss.shape) != 0:
            loss = loss.mean()
        return loss
