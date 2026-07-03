import torch
from torchvision.models import detr_resnet50, DETR_ResNet50_Weights


def get_model(num_classes):

    weights = DETR_ResNet50_Weights.DEFAULT

    model = detr_resnet50(
        weights=weights,
        num_classes=num_classes
    )

    return model