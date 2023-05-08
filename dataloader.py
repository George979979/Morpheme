import sys
from collections import defaultdict
import itertools

import torch
import numpy as np


def pad_tensor(vec, length, dim, pad_symbol):
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.shape[dim]
    answer = torch.cat([vec, torch.ones(*pad_size, dtype=torch.long) * pad_symbol], dim=dim)
    return answer

def pad_tensors(tensors, pad=0, dim=0, pad_inner=True):
    if isinstance(tensors[0], int):
        return torch.LongTensor(tensors)
    if dim > 0 and pad_inner:
        inner_tensors = [pad_tensors(tensor, pad=pad, dim=dim-1) for tensor in tensors]
        return pad_tensors(inner_tensors, pad=pad, dim=dim, pad_inner=False)
    tensor_type = torch.Tensor if "float" in str(getattr(tensors[0], "dtype", "")) else torch.LongTensor
    tensors = [tensor_type(tensor) for tensor in tensors]
    L = max(tensor.shape[dim] for tensor in tensors)
    tensors = [pad_tensor(tensor, L, dim=dim, pad_symbol=pad) for tensor in tensors]
    return torch.stack(tensors, dim=0)

class FieldBatchDataLoader:

    def __init__(self, X, batch_size=32, sort_by_length=True,
                 length_field=None, pad_dim=None,
                 state=115, device="cpu"):
        self.X = X
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.length_field = length_field
        self.pad_dim = pad_dim or dict()
        self.device = device
        np.random.seed(state)

    def __len__(self):
        return (len(self.X)-1) // self.batch_size + 1

    def __iter__(self):
        if self.sort_by_length:
            if self.length_field is not None:
                lengths = [len(x[self.length_field]) for x in self.X]
            else:
                lengths = [len(list(x.values())[0]) for x in self.X]
            order = np.argsort(lengths)
            batched_order = np.array([order[start:start+self.batch_size]
                                      for start in range(0, len(self.X), self.batch_size)])
            np.random.shuffle(batched_order[:-1])
            self.order = np.fromiter(itertools.chain.from_iterable(batched_order), dtype=int)
        else:
            self.order = np.arange(len(self.X))
            np.random.shuffle(self.order)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.X):
            raise StopIteration()
        end = min(self.idx + self.batch_size, len(self.X))
        indexes = [self.order[i] for i in range(self.idx, end)]
        batch = dict()
        batch_data = [self.X[i] for i in indexes]
        for field in self.X[indexes[0]]:
            data = [elem[field] for elem in batch_data]
            batch[field] = pad_tensors(data, dim=self.pad_dim.get(field, 0)).to(self.device)
        batch["indexes"] = indexes
        self.idx = end
        return batch

