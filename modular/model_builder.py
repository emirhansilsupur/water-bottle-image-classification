""" 
This module provides a function that creates a pre-trained image classification model with the specified architecture and number of output classes. 
Optionally, it freezes the feature weights and prints a summary of the model's architecture.
"""
import torch
import torchvision
from torchinfo import summary

def create_model(model_name:str, num_classes:int, freeze_features:bool=True, summary_model:bool=True):
    """
    Creates and returns a PyTorch pre-trained models based on the given model name and number of classes.

    Args:
    - model_name (str): The name of the model to create. It can be one of the following: "EfficientNet_B0", "ResNet18", or "GoogLeNet".
    - num_classes (int): The number of output classes.
    - freeze_features (bool, optional): If True, freezes the weights of the feature extractor layers of the model. Default is True.
    - summary_model (bool, optional): If True, prints a summary of the model. Default is True.

    Returns:
    - model (torch.nn.Module): The created PyTorch model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model based on the given model name
    if model_name == "EfficientNet_B0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        # Freeze the model features if specified
        if freeze_features:
            for param in weights.parameters():
                param.requires_grad = False
        # Update the last layer of the model for the specified number of classes
        num_ftrs = model._fc.in_features
        model._fc = torch.nn.Linear(num_ftrs, num_classes)
        

    elif model_name == "ResNet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights).to(device)
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

    elif model_name == "GoogLeNet":
        weights = torchvision.models.GoogLeNet_Weights.DEFAULT
        model = torchvision.models.googlenet(weights=weights).to(device)
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    # Print a summary of the model if summary_model is True
    if summary_model:
        summary(model=model, input_size=(3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])
    
    return model
