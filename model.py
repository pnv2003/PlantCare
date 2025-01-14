# finetune on light-weight CNN model
# MobileNetV2, SqueezeNet, EfficientNetB0, etc.
from typing import Literal
import torch
from torchvision import models

def CNN(
        model_name: Literal['mobilenet_v2', 'squeezenet1_1', 'efficientnet_b0', 'shufflenet_v2'],
        device="cpu",
        freeze=False,
        num_classes=10
    ):

    model = None
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    else:
        raise ValueError(f"Model {model_name} not supported yet")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if model_name == 'mobilenet_v2':
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'squeezenet1_1':
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif model_name == 'efficientnet_b0':
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'shufflenet_v2':
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    return model
