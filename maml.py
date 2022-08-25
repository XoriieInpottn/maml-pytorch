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

    def __init__(
            self,
            model: nn.Module, *,
            loss_fn: Callable,
            inner_lr: float,
            num_steps: int,
            first_order=False
    ) -> None:
        super(MAML, self).__init__()
        assert inner_lr > 0
        assert num_steps >= 1

        self.model = model
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.num_steps = num_steps
        self.first_order = first_order

        self.param_list = list(self.model.parameters())

        self._param_spec = {}
        memo = {id(p): p for p in self.param_list}
        self._model_symbol = copy.deepcopy(model, memo)
        self._make_param_spec(self._model_symbol)

        self._state_dict = None

    def _make_param_spec(self, module):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, nn.Parameter):
                delattr(module, name)
                self._param_spec[id(obj)] = (module, name)
        for child in module.children():
            self._make_param_spec(child)

    def forward(self, support_x, support_y, query_x, query_y):
        if query_x is None or query_y is None:
            return self.update(support_x, support_y)
        else:
            return self.compute_loss(support_x, support_y, query_x, query_y)

    def compute_loss(self, support_x, support_y, query_x, query_y):
        param_list = self.param_list
        for i in range(self.num_steps):
            pred_y = self.model(support_x) if i == 0 else self._model_symbol(support_x)
            loss = self.loss_fn(pred_y, support_y)
            if len(loss.shape) != 0:
                loss = loss.mean()

            new_param_list = []
            grad_list = autograd.grad(loss, param_list, create_graph=not self.first_order)
            if self.first_order:
                grad_list = [g.detach() for g in grad_list]

            for j in range(len(param_list)):
                new_param = param_list[j] - self.inner_lr * grad_list[j]
                new_param_list.append(new_param)
                module, name = self._param_spec[id(self.param_list[j])]
                setattr(module, name, new_param)
            param_list = new_param_list

        pred_y = self._model_symbol(query_x)
        loss = self.loss_fn(pred_y, query_y)
        if len(loss.shape) != 0:
            loss = loss.mean()
        return loss

    def update(self, support_x, support_y):
        pred = self.model(support_x)
        loss = self.loss_fn(pred, support_y)
        loss.backward()
        with torch.no_grad():
            for p in self.param_list:
                if not p.requires_grad or p.grad is None:
                    continue
                p.add_(p.grad, alpha=-self.inner_lr)
                p.grad = None
        return loss

    def checkpoint(self):
        self._state_dict = {
            name: value.clone().detach()
            for name, value in self.model.state_dict().items()
        }

    def restore(self):
        self.model.load_state_dict(self._state_dict)


class FOMAML(MAML):

    def __init__(
            self,
            model: nn.Module, *,
            loss_fn: Callable,
            inner_lr: float,
            num_steps: int
    ) -> None:
        super(FOMAML, self).__init__(
            model=model,
            loss_fn=loss_fn,
            inner_lr=inner_lr,
            num_steps=num_steps,
            first_order=True
        )
