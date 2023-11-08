from torch.utils.data import Sampler
import torch
import math


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group
        self.groups = self._group_by_length()

    def _group_by_length(self):
        lengths = torch.tensor(
            [
                self.data_source[i]["reference_audio"].shape[1]
                for i in range(len(self.data_source))
            ]
        )
        sorted_indices = lengths.argsort().tolist()
        groups = [
            sorted_indices[i : i + self.batches_per_group * self.batch_size]
            for i in range(
                0, len(sorted_indices), self.batches_per_group * self.batch_size
            )
        ]
        return groups

    def __iter__(self):
        group_indices = torch.randperm(len(self.groups)).tolist()
        batch_indices = []
        for i in group_indices:
            group = self.groups[i]
            shuffled_group = torch.randperm(len(group)).tolist()
            group = [group[i] for i in shuffled_group]
            batches = [
                group[i : i + self.batch_size]
                for i in range(0, len(group), self.batch_size)
            ]
            batch_indices.extend(batches)
        return iter(batch_indices)

    def __len__(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)
