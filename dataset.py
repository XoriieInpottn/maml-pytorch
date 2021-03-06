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

    def __init__(self,
                 ds_path,
                 num_ways,
                 num_shots,
                 transform_supp,
                 transform_query,
                 num_transforms=1):
        super(NKDataset, self).__init__()
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._transform_supp = transform_supp
        self._transform_query = transform_query
        self._num_transforms = num_transforms
        self._docs = collections.defaultdict(list)
        if isinstance(ds_path, str):
            ds_path = [ds_path]
        for ds_path_i in ds_path:
            with DocSet(ds_path_i, 'r') as ds:
                for doc in tqdm(ds, dynamic_ncols=True, leave=False, desc='Load data'):
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

        return (
            self._collate(support_task, self._transform_supp, self._num_transforms),
            self._collate(query_task, self._transform_query, 1)
        )

    @staticmethod
    def _collate(doc_list, transform, num_transforms):
        image_list = []
        label_list = []
        for doc in doc_list:
            image = doc['image']
            if isinstance(image, bytes):
                image = cv.imdecode(np.frombuffer(image, np.byte), cv.IMREAD_COLOR)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = np.array(doc['label'], np.int64)

            for _ in range(num_transforms):
                image_i = transform(image)
                image_i = np.array(image_i, np.float32)
                image_i = (image_i - 127.5) / 127.5
                image_i = np.transpose(image_i, (2, 0, 1))

                image_list.append(image_i)
                label_list.append(label)

        return {
            'image': np.stack(image_list),
            'label': np.stack(label_list)
        }
