from DLL_model import MyModel
import torch.nn as nn
import torch.optim as optim
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_html_components as html


import torch
class Training() : 
    
    def __init__(self, Xtrain, Ytrain, model_params) : 
        self.X = Xtrain
        self.Y = Ytrain

        self.model = MyModel(model_params)
        self.MSEloss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        self.epochs = 5000
        
        self.dis_epoch = []
        self.dis_loss = []
        
    def epochs(self, ep) : 
        self.epochs = ep
        
    
    def compute_global_loss(self, epoch) : 
        pred = self.model(torch.Tensor(self.X))
        loss = self.MSEloss(pred, torch.Tensor(self.Y))
        self.dis_epoch.append(epoch+1)
        self.dis_loss.append(loss.detach().numpy())
                
        return self.dis_loss
        
    def train(self) : 
        
        pred = self.model(torch.Tensor(self.X))
        
        loss = self.MSEloss(pred, torch.Tensor(self.Y))
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        
        
        


    
