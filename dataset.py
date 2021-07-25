#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-14
"""

import collections
import random

import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
from docset import DocSet
from torch.utils.data import IterableDataset
from tqdm import tqdm


class ImagenetTransform(object):

    def __init__(self, image_size: int, *, is_train: bool):
        self._augmenter = iaa.Sequential([
            iaa.Resize({'shorter-side': (image_size, int(image_size * 1.1)), 'longer-side': 'keep-aspect-ratio'}),
            iaa.Fliplr(0.5),
            iaa.Rotate((-10, 10), cval=127.5),
            iaa.CropToFixedSize(image_size, image_size),
            iaa.GaussianBlur((0.0, 0.1)),
            iaa.AddToBrightness((-10, 10)),
            iaa.AddToHue((-5, 5)),
        ]) if is_train else iaa.Sequential([
            iaa.Resize({'shorter-side': image_size, 'longer-side': 'keep-aspect-ratio'}),
            iaa.CenterCropToFixedSize(image_size, image_size),
        ])

    def __call__(self, image):
        return self._augmenter(image=image)


class NKDataset(IterableDataset):

    def __init__(self, ds_path, num_ways, num_shots, transform):
        super(NKDataset, self).__init__()
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._transform = transform
        self._docs = collections.defaultdict(list)
        if isinstance(ds_path, str):
            ds_path = [ds_path]
        for ds_path_i in ds_path:
            with DocSet(ds_path_i, 'r') as ds:
                for doc in tqdm(ds, leave=False):
                    label = doc['label']
                    self._docs[label].append(doc)
        self._docs = list(self._docs.values())

    def __getitem__(self, item):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        task = random.sample(self._docs, self._num_ways)
        task = [
            [{'image': doc['image'], 'label': label} for doc in random.sample(sample_list, self._num_shots * 2)]
            for label, sample_list in enumerate(task)
        ]

        support_task = []
        query_task = []
        for sample_list in task:
            support_task.extend(sample_list[:self._num_shots])
            query_task.extend(sample_list[self._num_shots:])

        return self._collate(support_task), self._collate(query_task)

    def _collate(self, doc_list):
        image_list = []
        label_list = []
        for doc in doc_list:
            image = doc['image']
            if isinstance(image, bytes):
                image = cv.imdecode(np.frombuffer(image, np.byte), cv.IMREAD_COLOR)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            if callable(self._transform):
                image = self._transform(image)

            image = np.array(image, np.float32)
            image = (image - 127.5) / 127.5
            image = np.transpose(image, (2, 0, 1))
            image_list.append(image)

            label_list.append(np.array(doc['label'], np.int64))

        return {
            'image': np.stack(image_list),
            'label': np.stack(label_list)
        }
