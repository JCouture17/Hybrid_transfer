### Load Model ###
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import torch

class transfer_model:
    
    def load_model(model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        elif model_name == 'vgg11':
            model = models.vgg11(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False 
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1))
        
        summary(model,(3,224,224))
        if torch.cuda.is_available:
            model.cuda()
        return model

