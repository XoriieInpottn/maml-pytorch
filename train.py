#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-06-29
"""

import argparse
import os

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

import dataset
import utils
from maml import MAML


class Trainer(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use.')
        parser.add_argument('--train-path', required=True, help='Path of the directory that contains the data files.')
        parser.add_argument('--test-path', required=True, help='Path of the directory that contains the data files.')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
        parser.add_argument('--num-loops', type=int, default=2000, help='The number of loops to train.')
        parser.add_argument('--max-lr', type=float, default=1e-3, help='The maximum value of learning rate.')
        parser.add_argument('--weight-decay', type=float, default=0.3, help='The weight decay value.')
        parser.add_argument('--optimizer', default='AdamW', help='Name of the optimizer to use.')

        parser.add_argument('--num-ways', type=int, default=5)
        parser.add_argument('--num-shots', type=int, default=5)
        parser.add_argument('--inner-lr', type=float, default=0.1)
        parser.add_argument('--num-steps', type=int, default=3)
        parser.add_argument('--output-dir', default='output')
        self._args = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = self._args.gpu

        self._device = 'cpu' if self._args.gpu == 'cpu' else 'cuda'

        # create dataset and data loader
        self._train_dataset = dataset.NKDataset(self._args.train_path, self._args.num_ways, self._args.num_shots)
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._args.batch_size,
            num_workers=5,
            pin_memory=True
        )

        self._test_dataset = dataset.NKDataset(self._args.test_path, self._args.num_ways, self._args.num_shots)
        self._test_loader = DataLoader(
            self._test_dataset,
            batch_size=self._args.batch_size,
            num_workers=5,
            pin_memory=True
        )

        # init model
        self._model = resnet18().to(self._device)
        self._loss = nn.CrossEntropyLoss()
        self._maml = MAML(
            model=self._model,
            loss_fn=self._loss,
            lr=self._args.inner_lr,
            num_steps=self._args.num_steps
        )

        # optimizer
        self._parameters = list(self._model.parameters())
        optimizer_class = getattr(optim, self._args.optimizer)
        self._optimizer = optimizer_class(
            self._parameters,
            lr=self._args.max_lr,
            weight_decay=self._args.weight_decay
        )
        self._scheduler = utils.CosineWarmUpAnnealingLR(self._optimizer, self._args.num_loops)
        self._inner_optimizer = optim.SGD(self._parameters, lr=self._args.inner_lr)

    def train(self):
        loss_g = 0.0
        it = iter(self._train_loader)
        loops = tqdm(range(self._args.num_loops))

        for loop in loops:
            support_doc, query_doc = next(it)
            support_image = support_doc['image']
            support_mask = support_doc['mask']
            query_image = query_doc['image']
            query_mask = query_doc['mask']

            loss, lr = self._train_step(support_image, support_mask, query_image, query_mask)
            loss = float(loss.numpy())
            loss_g = 0.9 * loss_g + 0.1 * loss
            loops.set_description(f'[{loop + 1}/{self._args.num_loops}] L={loss_g:.06f} lr={lr:.01e}', False)

            if (loop + 1) % 100 == 0 or (loop + 1) == self._args.num_loops:
                precision, recall = self.evaluate(10)
                loops.write(
                    f'[{loop + 1}/{self._args.num_loops}] '
                    f'L={loss_g:.06f} '
                    f'precision={precision:.02%} '
                    f'recall={recall:.02%}'
                )
                self._save_model()

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
        for sx, sy, qx in zip(support_x, support_y, query_x):
            state_dict = self._model.state_dict()
            for i in range(self._args.num_steps):
                pred = self._model(sx)
                true = F.one_hot(sy, self._args.num_ways)
                loss = self._loss(pred, true)
                loss.backward()
                self._inner_optimizer.step()
                self._inner_optimizer.zero_grad()
            pred = self._model(qx)
            qy = torch.argmax(pred, 1)
            qy_list.append(qy)
            self._model.load_state_dict(state_dict)
        query_y = torch.stack(qy_list)
        return query_y.detach().cpu()

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
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]


if __name__ == '__main__':
    raise SystemExit(Trainer().train())
