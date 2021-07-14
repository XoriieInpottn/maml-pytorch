#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-14
"""

import numpy as np
import cv2 as cv
import collections
import random

import torch
from docset import DocSet
from torch.utils.data import IterableDataset
from tqdm import tqdm


class NKDataset(IterableDataset):

    def __init__(self, ds_path, num_ways=5, num_shots=5):
        super(NKDataset, self).__init__()
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._docs = collections.defaultdict(list)
        with DocSet(ds_path, 'r') as ds:
            for doc in tqdm(ds):
                label = doc['label']
                self._docs[label].append(doc)
        self._docs = list(self._docs.values())

    def __getitem__(self, item):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        selected_docs = [
            {'image': doc['image'], 'label': label}
            for label, clazz_set in enumerate(random.sample(self._docs, self._num_ways))
            for doc in random.sample(clazz_set, self._num_shots)
        ]
        image_list = []
        label_list = []
        for doc in selected_docs:
            image = cv.imdecode(np.frombuffer(doc['image'], np.byte), cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (224, 224))
            image = torch.from_numpy(image)
            image = image.float() / 127.5 - 1.0
            image = image.permute((2, 0, 1))
            image_list.append(image)

            label_list.append(torch.tensor(doc['label'], dtype=torch.int64))

        return {
            'image': torch.stack(image_list),
            'label': torch.stack(label_list)
        }
