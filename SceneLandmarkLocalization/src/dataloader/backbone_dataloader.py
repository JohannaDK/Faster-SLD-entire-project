from torch.utils.data import DataLoader,Sampler,Dataset
import numpy as np
import random

class CombinedDataset(Dataset):
    def __init__(self, dataset_list, shuffle):
        self.dataset_list = dataset_list
        self.dataset_sizes = [len(dataset) for dataset in dataset_list]
        self.total_size = sum(self.dataset_sizes)
        self.cumulative_sizes = [0] + list(np.cumsum(self.dataset_sizes))

        # to shuffle dataset within each scene
        # if shuffle:
        #     for ds in self.dataset_list:
        #         zipped = list(zip(*ds.values()))

        #         random.shuffle(zipped)

        #         ds = {key: list(value) for key, value in zip(data.keys(), zip(*zipped))}
                

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i, dataset_cumulative_size in enumerate(self.cumulative_sizes):
            if idx < dataset_cumulative_size:
                dataset_idx = i - 1
                sample_idx = idx - self.cumulative_sizes[dataset_idx]
                return self.dataset_list[dataset_idx][sample_idx]

class HomogeneousBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.current_idx = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = []
        cur_i = 0
        # shuffle batches as well, so each iteration of training loop we train on a random scene
        while cur_i < len(self.dataset):
            for cumsum in self.dataset.cumulative_sizes:
                if cur_i < cumsum:
                    break
            batches.append(list(range(cur_i, min(cur_i + self.batch_size, cumsum))))
            if cur_i + self.batch_size < cumsum:
                cur_i += self.batch_size
            else:
                cur_i = cumsum
        if self.shuffle:
            random.shuffle(batches)
        while self.current_idx < len(batches):
            yield batches[self.current_idx]
            self.current_idx += 1

    def __len__(self):
        # Total number of batches across all datasets
        return len(self.dataset) // self.batch_size