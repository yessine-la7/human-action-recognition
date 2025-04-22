import torch.nn as nn
from torchvision import models

def create_custom_model(num_classes, use_pretrained=True):
    # Load pre-trained ResNet-50 model
    if use_pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    # Get the base ResNet-50 model
    base_model = models.resnet50(weights=weights)
    #base_model = models.SqueezeNet(weights=weights)

    #freeze the layers of ResNet-50
    for param in base_model.parameters():
        param.requires_grad = False

    # Remove the fully connected layer of ResNet-50 (model.fc)
    # We will use the output from the last convolutional layer
    modules = list(base_model.children())[:-2]  # Remove the last2 layers fc+pool 
    base_model = nn.Sequential(*modules)

    # new head with additional convolutional layers and linear layers
    class CustomModel(nn.Module):
        def __init__(self, base_model, num_classes):
            super(CustomModel, self).__init__()
            self.base_model = base_model
            # Additional convolutional layers
            self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)  # 2048 input channels from ResNet-50
            self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # Final fully connected layer
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            # Pass through base ResNet-50 (except the final fc layer)
            x = self.base_model(x)
            x = x.view(x.size(0), 2048, 7, 7)  # Reshape for the new conv layers
            
            # Pass through additional convolutional layers
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            
            # Global average pooling
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            
            # Pass through final fully connected layer
            x = self.fc(x) # raw logit output
            return x

    # Create an instance of the custom model
    model = CustomModel(base_model, num_classes)
    return model
