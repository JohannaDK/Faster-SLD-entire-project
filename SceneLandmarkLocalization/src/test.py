import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

"""
testing out how to specify the optimizer
"""

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv = nn.Conv2d(1,2,3)

    def forward(self,x):
        return self.conv(x)


class Head(nn.Module):
    def __init__(self):
        super(Head,self).__init__()
        self.conv = nn.Conv2d(2,1,1)
    
    def forward(self,x):
        return self.conv(x)

# 3 training samples, batch size 1, 1 channel, 5x5 data
data = np.random.rand(3,1,1,5,5)
labels = np.random.rand(3,1,1,3,3)

bb = Backbone()
scenes = ["scene1","scene2","scene3"]
heads = {}
for s in scenes:
    heads[s] = Head()

models = {}
for s in scenes:
    models[s] = nn.Sequential(bb, heads[s])

optimizers = {}
for s in scenes:
    optimizers[s] = optim.SGD(models[s].parameters(), lr = 1e-3)

for epoch in range(2):
    for i, d in enumerate(data):
        print("head: ", i)
        print("epoch: ", epoch)
        scene = scenes[i]
        for name, param in models[scene].named_parameters():
            print(name, param.data)
        input = torch.tensor(d,dtype=torch.float32)
        label = torch.tensor(labels[i],dtype=torch.float32)
        optimizers[scene].zero_grad()
        pred = models[scene](input)
        loss = torch.sum((pred-label)**2)
        loss.backward()
        optimizers[scene].step()
        for name, param in models[scene].named_parameters():
            print(name, param.data)

"""
this worked, backbone is updated every iteration but each head only when it's passed through
"""