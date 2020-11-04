import torch
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, params):
        super(MyModel, self).__init__()
        
        self.params = params
        le = len(self.params)
        
        self.feature = nn.Sequential()
        for i in range(le-2) : 
            self.feature.add_module("hdn_{}".format(i+1), nn.Linear(self.params[i], self.params[i+1]))
            self.feature.add_module("relu_{}".format(i+1), nn.ELU())
        self.feature.add_module("hdn_{}".format(le-1), nn.Linear(self.params[le-2], self.params[le-1]))
        
    def forward(self, x):
        x = x.reshape(-1, 1)
        return self.feature(x).squeeze()