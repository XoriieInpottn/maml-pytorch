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
from torch.utils.data import Dataset
from tqdm import tqdm


class NKDataset(Dataset):

    def __init__(
            self,
            ds_path,
            num_ways,
            num_shots,
            transform_supp,
            transform_query,
            size,
            times_of_query_samples=1
    ):
        super(NKDataset, self).__init__()
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._transform_supp = transform_supp
        self._transform_query = transform_query
        self._size = size
        self._times_of_query_samples = times_of_query_samples
        self._docs = collections.defaultdict(list)
        if isinstance(ds_path, str):
            ds_path = [ds_path]
        for ds_path_i in ds_path:
            with DocSet(ds_path_i, 'r') as ds:
                for doc in tqdm(ds, dynamic_ncols=True, leave=False, desc='Load data'):
                    label = doc['label']
                    self._docs[label].append(doc)
        self._docs = list(self._docs.values())

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        task = random.sample(self._docs, self._num_ways)
        task = [
            [{'image': doc['image'], 'label': label}
             for doc in random.sample(sample_list, self._num_shots * (1 + self._times_of_query_samples))]
            for label, sample_list in enumerate(task)
        ]  # k samples for support set, ak samples for query set

        support_task = []
        query_task = []
        for sample_list in task:
            support_task.extend(sample_list[:self._num_shots])
            query_task.extend(sample_list[self._num_shots:])

        return (
            self._collate(support_task, self._transform_supp),
            self._collate(query_task, self._transform_query)
        )

    @staticmethod
    def _collate(doc_list, transform):
        image_list = []
        label_list = []
        for doc in doc_list:
            image = doc['image']
            if isinstance(image, bytes):
                image = cv.imdecode(np.frombuffer(image, np.byte), cv.IMREAD_COLOR)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = np.array(doc['label'], np.int64)

            image_i = transform(image=image)
            image_i = np.array(image_i, np.float32)
            image_i = (image_i - 127.5) / 127.5
            image_i = np.transpose(image_i, (2, 0, 1))

            image_list.append(image_i)
            label_list.append(label)

        return {
            'image': np.stack(image_list),
            'label': np.stack(label_list)
        }


class TrainDataset(NKDataset):

    def __init__(
            self,
            ds_path,
            image_size,
            num_ways,
            num_shots,
            size
    ) -> None:
        transform = iaa.Sequential([
            iaa.Resize((image_size, image_size), interpolation='linear'),
            iaa.Fliplr(0.5)
        ])
        super(TrainDataset, self).__init__(
            ds_path=ds_path,
            num_ways=num_ways,
            num_shots=num_shots,
            transform_supp=transform,
            transform_query=transform,
            size=size,
            times_of_query_samples=1
        )


class TestDataset(NKDataset):

    def __init__(
            self,
            ds_path,
            image_size,
            num_ways,
            num_shots,
            size
    ) -> None:
        transform = iaa.Resize((image_size, image_size), interpolation='linear')
        super(TestDataset, self).__init__(
            ds_path=ds_path,
            num_ways=num_ways,
            num_shots=num_shots,
            transform_supp=transform,
            transform_query=transform,
            size=size,
            times_of_query_samples=3
        )
