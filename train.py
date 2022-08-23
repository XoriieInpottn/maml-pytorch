#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-06-29
"""

import argparse
import os
from typing import Iterable

import cv2 as cv
import numpy as np
import torch
from sklearn import metrics
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model
import utils
from maml import MAML

cv.setNumThreads(0)


def scale_grad_by_value(params: Iterable[nn.Parameter], max_value: float) -> None:
    for p in params:
        if p.requires_grad and p.grad is not None:
            grad_max = float(p.grad.abs().max())
            if grad_max > max_value:
                p.grad.mul_(max_value / grad_max)


class Trainer(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-path', required=True)
        parser.add_argument('--batch-size', type=int, default=2)
        parser.add_argument('--num-epochs', type=int, default=200)
        parser.add_argument('--max-lr', type=float, default=1e-3)
        parser.add_argument('--weight-decay', type=float, default=1e-4)
        parser.add_argument('--optimizer', default='SGD')

        parser.add_argument('--image-size', type=int, default=84)
        parser.add_argument('--ch-hid', type=int, default=64)
        parser.add_argument('--num-ways', type=int, default=5)
        parser.add_argument('--num-shots', type=int, default=5)
        parser.add_argument('--inner-lr', type=float, default=1e-2)
        parser.add_argument('--num-steps', type=int, default=5)
        parser.add_argument('--first-order', action='store_true')
        parser.add_argument('--train-size', type=int, default=1000)
        parser.add_argument('--test-size', type=int, default=100)
        parser.add_argument('--output-dir', default='output')
        self._args = parser.parse_args()

        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            raise RuntimeError('You must set environment variable CUDA_VISIBLE_DEVICES.')
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._create_dataset()
        self._create_model()
        self._create_optimizer()

    def _create_dataset(self):
        self._train_dataset = dataset.NKDataset(
            [os.path.join(self._args.data_path, 'train.ds'),
             os.path.join(self._args.data_path, 'valid.ds')],
            num_ways=self._args.num_ways,
            num_shots=self._args.num_shots,
            transform_supp=dataset.ImagenetTransform(self._args.image_size, is_train=True),
            transform_query=dataset.ImagenetTransform(self._args.image_size, is_train=True),
            size=self._args.train_size,
            # num_transforms=3
        )
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._args.batch_size,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=8,
            pin_memory=True,
        )
        self._test_dataset = dataset.NKDataset(
            os.path.join(self._args.data_path, 'test.ds'),
            num_ways=self._args.num_ways,
            num_shots=self._args.num_shots,
            transform_supp=dataset.ImagenetTransform(self._args.image_size, is_train=False),
            transform_query=dataset.ImagenetTransform(self._args.image_size, is_train=False),
            size=self._args.test_size,
            # num_transforms=3
        )
        self._test_loader = DataLoader(
            self._test_dataset,
            batch_size=self._args.batch_size,
            num_workers=8,
            pin_memory=True
        )

    def _create_model(self):
        self._model = model.Model(
            self._args.image_size,
            num_classes=self._args.num_ways,
            ch_hid=self._args.ch_hid
        ).to(self._device)
        self._loss_fn = nn.CrossEntropyLoss()
        self._maml = MAML(
            self._model,
            loss_fn=self._loss_fn,
            inner_lr=self._args.inner_lr,
            num_steps=self._args.num_steps,
            first_order=self._args.first_order
        )

    def _create_optimizer(self):
        self._parameters = list(self._model.parameters())
        optimizer_class = getattr(optim, self._args.optimizer)
        opt_args = {
            'params': self._parameters,
            'lr': self._args.max_lr,
            'weight_decay': self._args.weight_decay,
            'momentum': 0.9,  # SGD, RMSprop
            'betas': (0.9, 0.999),  # Adam*
        }
        co = optimizer_class.__init__.__code__
        self._optimizer = optimizer_class(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })
        num_loops = self._args.num_epochs * len(self._train_loader)
        self._scheduler = utils.CosineWarmUpAnnealingLR(self._optimizer, num_loops)

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

    def _predict_step(self, support_x, support_y, query_x):
        """Predict a batch of tasks. Each task is consist of several samples.

        Args:
            support_x (torch.Tensor): (batch_size, num_samples, ...)
            support_y (torch.Tensor): (batch_size, num_samples, ...)
            query_x (torch.Tensor): (batch_size, num_samples, ...)

        Returns:
            torch.Tensor: query_y (batch_size, num_samples, ...)
        """
        support_x = support_x.to(self._device)
        support_y = support_y.to(self._device)
        query_x = query_x.to(self._device)

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
        support_x = support_x.to(self._device)
        support_y = support_y.to(self._device)
        query_x = query_x.to(self._device)
        query_y = query_y.to(self._device)

        loss = self._maml(support_x, support_y, query_x, query_y)
        loss.backward()
        scale_grad_by_value(self._model.parameters(), 0.1)
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]

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


if __name__ == '__main__':
    raise SystemExit(Trainer().train())
