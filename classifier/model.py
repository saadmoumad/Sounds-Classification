import torch
import torch.nn as nn
from torchvision import models
import io
from torchvision.transforms import ToTensor
import os
from torchsummary import summary

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, resnet.conv1.out_channels,
                                         kernel_size = resnet.conv1.kernel_size[0],
                                         stride = resnet.conv1.stride[0],
                                         padding = resnet.conv1.padding[0])
       
        modules = list(resnet.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        
        #self.linear = nn.Linear(resnet.fc.in_features, 2)
        self.fc = nn.Linear(resnet.fc.in_features, 8)
       
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 2)
        
        self.softmax = nn.LogSoftmax()
        self.bn2 = nn.BatchNorm1d(2, momentum=0.01)
        self.bn8 = nn.BatchNorm1d(8, momentum=0.01)
        
        for param in self.base_model.parameters():
                param.requires_grad = False        
        
        #self.classifier = nn.Sequential(self.relu, self.output, self.sigmoid )
        
        #self.base_model.fc = nn.Sequential(nn.Dropout(p=configuration_dict.get('dropout', 0.25)), 
         #                                    self.fc)''
        
    def forward(self, x):
        with torch.no_grad():
            x = self.base_model(x)
                
        x = x.reshape(x.size(0), -1)
        x = self.bn8(self.fc(x))
        x = self.relu(x)
        x = self.bn2(self.output(x))
        x = self.softmax(x)
        return x