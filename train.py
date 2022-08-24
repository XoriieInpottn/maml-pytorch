#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-06-29
"""

import argparse
import os
from math import inf
from typing import Callable

import cv2 as cv
import numpy as np
import torch
from sklearn import metrics
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import TrainDataset, TestDataset
from maml import MAML
from model import Model

cv.setNumThreads(0)


class Config(object):

    def __init__(self):
        self.train_path = None
        self.test_path = None

        self.train_size = 1000
        self.test_size = 100

        self.image_size = 84
        self.num_ways = 5
        self.num_shots = 5

        self.batch_size = 4
        self.num_epochs = 100
        self.max_lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 0.3
        self.optimizer = 'AdamW'

        self.inner_lr = 1e-2
        self.num_steps = 5
        self.first_order = False

        self.device = 'cpu'

    def load(self, args):
        for name in dir(self):
            if name.startswith('_') or callable(getattr(self, name)):
                continue
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))


class Trainer(object):

    def __init__(
            self,
            model: nn.Module,
            loss_fn: Callable,
            config: Config
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._args = config

        self._model = model.to(self._args.device)
        if isinstance(self._loss_fn, nn.Module):
            self._loss_fn.to(self._args.device)

        self._maml = MAML(
            self._model,
            loss_fn=self._loss_fn,
            inner_lr=self._args.inner_lr,
            num_steps=self._args.num_steps,
            first_order=self._args.first_order
        )

        self._create_dataset()
        self._create_optimizer()

    def _create_dataset(self):
        self._train_loader = DataLoader(
            TrainDataset(
                self._args.train_path,
                image_size=self._args.image_size,
                num_ways=self._args.num_ways,
                num_shots=self._args.num_shots,
                size=self._args.train_size
            ),
            batch_size=self._args.batch_size,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=8,
            pin_memory=True,
        )
        self._test_loader = DataLoader(
            TestDataset(
                self._args.test_path,
                image_size=self._args.image_size,
                num_ways=self._args.num_ways,
                num_shots=self._args.num_shots,
                size=self._args.test_size
            ),
            batch_size=self._args.batch_size,
            num_workers=8,
            pin_memory=True
        )

    def _create_optimizer(self):
        self._parameters = list(self._model.parameters())
        optimizer_class = getattr(optim, self._args.optimizer)
        opt_args = {
            'params': self._parameters,
            'lr': self._args.max_lr,
            'weight_decay': self._args.weight_decay,
            'momentum': self._args.momentum,  # SGD, RMSprop
            'betas': (self._args.momentum, 0.999),  # Adam*
        }
        co = optimizer_class.__init__.__code__
        self._optimizer = optimizer_class(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })
        num_loops = self._args.num_epochs * len(self._train_loader)
        self._scheduler = utils.CosineWarmUpAnnealingLR(self._optimizer, num_loops)

    def _train_step(self, support_x, support_y, query_x, query_y):
        """Train with a batch of tasks. Each task is consist of several samples.

        Args:
            support_x (torch.Tensor): (batch_size, num_samples, ...)
            support_y (torch.Tensor): (batch_size, num_samples, ...)
            query_x (torch.Tensor): (batch_size, num_samples, ...)
            query_y (torch.Tensor): (batch_size, num_samples, ...)

        Returns:
            tuple[torch.Tensor, int]: loss and the current learning rate
        """
        support_x = support_x.to(self._args.device)
        support_y = support_y.to(self._args.device)
        query_x = query_x.to(self._args.device)
        query_y = query_y.to(self._args.device)

        loss = self._maml(support_x, support_y, query_x, query_y)
        loss.backward()
        clip_grad_norm_(self._model.parameters(), 0.1, inf)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]

    def _predict_step(self, support_x, support_y, query_x):
        """Predict a batch of tasks. Each task is consist of several samples.

        Args:
            support_x (torch.Tensor): (batch_size, num_samples, ...)
            support_y (torch.Tensor): (batch_size, num_samples, ...)
            query_x (torch.Tensor): (batch_size, num_samples, ...)

        Returns:
            torch.Tensor: query_y (batch_size, num_samples, ...)
        """
        support_x = support_x.to(self._args.device)
        support_y = support_y.to(self._args.device)
        query_x = query_x.to(self._args.device)

        qy_list = []
        self._maml.checkpoint()
        for sx, sy, qx in zip(support_x, support_y, query_x):
            for i in range(self._args.num_steps * 2):
                pred = self._model(sx)
                loss = self._loss_fn(pred, sy)
                loss.backward()
                with torch.no_grad():
                    for p in self._model.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        p.add_(p.grad, alpha=-self._args.inner_lr)
                        p.grad = None

            pred = self._model(qx)
            qy = torch.argmax(pred, 1)
            qy_list.append(qy)

            self._maml.restore()

        return torch.stack(qy_list).detach().cpu()

    def train(self):
        loss_g = None
        for epoch in range(self._args.num_epochs):
            with tqdm(total=len(self._train_loader), leave=False, ncols=96) as loop:
                for support_doc, query_doc in self._train_loader:
                    loop.update()
                    support_x = support_doc['image']
                    support_y = support_doc['label']
                    query_x = query_doc['image']
                    query_y = query_doc['label']

                    loss, lr = self._train_step(support_x, support_y, query_x, query_y)
                    loss = float(loss.numpy())
                    loss_g = 0.9 * loss_g + 0.1 * loss if loss_g is not None else loss
                    loop.set_description(
                        f'[{epoch + 1}/{self._args.num_epochs}] '
                        f'L={loss_g:.06f} '
                        f'lr={lr:.01e}',
                        False
                    )

            acc = self._evaluate()
            print(
                f'[{epoch + 1}/{self._args.num_epochs}] '
                f'L={loss_g:.06f} '
                f'acc={acc:.02%} '
            )

    def _evaluate(self):
        true_list = []
        pred_list = []
        with tqdm(total=len(self._test_loader), leave=False, ncols=96) as loop:
            for support_doc, query_doc in self._test_loader:
                loop.update()
                support_x = support_doc['image']
                support_y = support_doc['label']
                query_x = query_doc['image']
                query_y = query_doc['label']
                pred_y = self._predict_step(support_x, support_y, query_x)
                true = query_y.numpy().reshape((-1,)).astype(np.int64)
                pred = pred_y.numpy().reshape((-1,)).astype(np.int64)
                true_list.extend(true)
                pred_list.extend(pred)
        return metrics.accuracy_score(true_list, pred_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--optimizer', default='AdamW')

    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--ch-hid', type=int, default=32)
    parser.add_argument('--num-ways', type=int, default=5)
    parser.add_argument('--num-shots', type=int, default=5)
    parser.add_argument('--inner-lr', type=float, default=1e-2)
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--test-size', type=int, default=100)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    config = Config()
    config.load(args)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError('You must set environment variable CUDA_VISIBLE_DEVICES.')
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.train_path = [os.path.join(args.data_path, 'train.ds'), os.path.join(args.data_path, 'valid.ds')]
    config.test_path = os.path.join(args.data_path, 'test.ds')

    model = Model(
        args.image_size,
        num_classes=args.num_ways,
        ch_hid=args.ch_hid
    )
    loss_fn = nn.CrossEntropyLoss()

    Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config
    ).train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
