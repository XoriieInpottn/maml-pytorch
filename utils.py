#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-06-29
"""
import io
import math

from torch.optim.lr_scheduler import LambdaLR


class BaseConfig(object):

    def load(self, args):
        for name, value in self.__dict__.items():
            if name.startswith('_') or callable(value):
                continue
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))

    def __str__(self):
        pairs = []
        max_len = 0
        for name, value in self.__dict__.items():
            if name.startswith('_') or callable(value):
                continue
            if value is None:
                value = 'None'
            elif isinstance(value, str):
                value = f'\'{value}\''
            else:
                value = str(value)
            l = len(name)
            if l > max_len:
                max_len = l
            pairs.append((name, value))

        buffer = io.StringIO()
        for name, value in pairs:
            buffer.write(name.rjust(max_len))
            buffer.write(': ')
            buffer.write(value)
            buffer.write('\n')
        return buffer.getvalue()


class CosineWarmUpAnnealingLR(LambdaLR):

    def __init__(self,
                 optimizer,
                 num_loops,
                 warm_up_proportion=0.01,
                 max_factor=1.0,
                 min_factor=0.0,
                 pow_warm_up=None,
                 pow_annealing=2.0,
                 last_epoch=-1):
        self._num_loops = num_loops
        self._warm_up_proportion = warm_up_proportion
        self._max_factor = max_factor
        self._min_factor = min_factor
        self._pow_warm_up = pow_warm_up
        self._pow_annealing = pow_annealing
        self._warm_up_loops = int(self._warm_up_proportion * self._num_loops)
        super(CosineWarmUpAnnealingLR, self).__init__(
            optimizer=optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=last_epoch
        )

    def _lr_lambda(self, i: int) -> float:
        if self._warm_up_loops == 0:
            return self._max_factor
        if i <= self._warm_up_loops:
            i = i - self._warm_up_loops + 1
            value = (math.cos(i * math.pi / self._warm_up_loops) + 1.0) * 0.5
            if self._pow_warm_up is not None and self._pow_warm_up != 1.0:
                value = math.pow(value, self._pow_warm_up)
            value = value * (self._max_factor - self._min_factor) + self._min_factor
        else:
            if i >= self._num_loops:
                i = self._num_loops - 1
            i = i - self._warm_up_loops
            value = (math.cos(i * math.pi / (self._num_loops - self._warm_up_loops)) + 1.0) * 0.5
            if self._pow_annealing is not None and self._pow_annealing != 1.0:
                value = math.pow(value, self._pow_annealing)
            value = value * (self._max_factor - self._min_factor) + self._min_factor
        return value
