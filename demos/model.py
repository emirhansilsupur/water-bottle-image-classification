import torch
import torchvision

from torch import nn


def create_model(num_classes: int = 3):
    """
    Creates and returns a PyTorch pre-trained models based on the given model name and number of classes.

    Args:
    - num_classes (int): The number of output classes.


    Returns:
    - model (torch.nn.Module): The created PyTorch model.
    """

    # Set the seed for general torch operations
    torch.manual_seed(42)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(42)

    # Create model

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model_transforms = weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=weights)
    # Freeze the model features if specified

    for param in model.features.parameters():
        param.requires_grad = False
    # Update the last layer of the model for the specified number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.25, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=3,  # same number of output units as our number of classes
        ),
    )
    return model, model_transforms
