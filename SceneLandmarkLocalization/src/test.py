import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

"""
testing out how to specify the optimizer
"""

# class Backbone(nn.Module):
#     def __init__(self):
#         super(Backbone, self).__init__()
#         self.conv = nn.Conv2d(1,2,3)

#     def forward(self,x):
#         return self.conv(x)


# class Head(nn.Module):
#     def __init__(self):
#         super(Head,self).__init__()
#         self.conv = nn.Conv2d(2,1,1)
    
#     def forward(self,x):
#         return self.conv(x)

# # 3 training samples, batch size 1, 1 channel, 5x5 data
# data = np.random.rand(3,1,1,5,5)
# labels = np.random.rand(3,1,1,3,3)

# bb = Backbone()
# scenes = ["scene1","scene2","scene3"]
# heads = {}
# for s in scenes:
#     heads[s] = Head()

# models = {}
# for s in scenes:
#     models[s] = nn.Sequential(bb, heads[s])

# optimizers = {}
# for s in scenes:
#     optimizers[s] = optim.SGD(models[s].parameters(), lr = 1e-3)

# for epoch in range(2):
#     for i, d in enumerate(data):
#         print("head: ", i)
#         print("epoch: ", epoch)
#         scene = scenes[i]
#         for name, param in models[scene].named_parameters():
#             print(name, param.data)
#         input = torch.tensor(d,dtype=torch.float32)
#         label = torch.tensor(labels[i],dtype=torch.float32)
#         optimizers[scene].zero_grad()
#         pred = models[scene](input)
#         loss = torch.sum((pred-label)**2)
#         loss.backward()
#         optimizers[scene].step()
#         for name, param in models[scene].named_parameters():
#             print(name, param.data)

"""
this worked, backbone is updated every iteration but each head only when it's passed through
"""

"""
testing dataloader
"""

from torch.utils.data import DataLoader,Sampler,Dataset
import random

class CombinedDataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.dataset_sizes = [len(dataset) for dataset in dataset_list]
        self.total_size = sum(self.dataset_sizes)
        self.cumulative_sizes = [0] + list(np.cumsum(self.dataset_sizes))

        for ds in self.dataset_list:
            random.shuffle(ds)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i, dataset_cumulative_size in enumerate(self.cumulative_sizes):
            if idx < dataset_cumulative_size:
                dataset_idx = i - 1
                sample_idx = idx - self.cumulative_sizes[dataset_idx]
                return self.dataset_list[dataset_idx][sample_idx]

class HomogeneousBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.current_idx = 0
        self.batch_size = batch_size

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
        random.shuffle(batches)
        while self.current_idx < len(batches):
            yield batches[self.current_idx]
            self.current_idx += 1

        

    def __len__(self):
        # Total number of batches across all datasets
        return len(self.dataset) // self.batch_size
  
ds1 = [{"data":x,"dataset":y} for (x,y) in list(zip(np.random.rand(10),[1]*10))]
ds2 = [{"data":x,"dataset":y} for (x,y) in list(zip(np.random.rand(6),[2]*6))]
ds3 = [{"data":x,"dataset":y} for (x,y) in list(zip(np.random.rand(9),[3]*9))]
print(ds1)
datasets = CombinedDataset([ds1,ds2,ds3])
batch_size = 2
sampler = HomogeneousBatchSampler(datasets, batch_size)
dataloader = DataLoader(datasets, batch_sampler=sampler)

for batch in dataloader:
    print(batch)
