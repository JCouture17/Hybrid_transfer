### Load Model ###
import torch.nn as nn
from torchvision import models
import torch
from torchsummary import summary

class transfer_model:
    
    def load_model(model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)

        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)

        elif model_name == 'vgg11':
            model = models.vgg11(pretrained=True)

        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True)

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)

        elif model_name == 'efficientnet':
            model = models.efficientnet_b7(pretrained=True)
            
        elif model_name == 'densenet':
            model = models.densenet161(pretrained=True)
            
        elif model_name == 'regnet':
            model = models.regnet_y_32gf(pretrained=True)
            
        for param in model.parameters():
            param.requires_grad=False
            
        # summary(model, (3,224,224))
            
        try:
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1))
        # except:
        #     num_features = model.classifier.in_features
        #     # print("didn't get to first")
        except AttributeError:
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1))
        except TypeError:
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1))   
        except:
            num_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1))       
            
            
            # summary(model, (3,224,244))
        
        if torch.cuda.is_available:
            model.cuda()
        return model

