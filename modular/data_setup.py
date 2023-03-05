"""
Modular Data Setup

This module contains functions and classes for setting up data for machine learning models.

Functions:
- find_classes(data) -> Tuple[List[str], Dict[str, int]]: Finds class names by scanning the target directory
- create_dataloaders(train_dir: str,test_dir: str,transform: transforms.Compose,batch_size: int) -> Tuple[DataLoader, DataLoader, List[str]]: Creates and returns train and test dataloaders along with class names

Classes:
- CustomImageFolder(Dataset): Custom Image folder dataset class

"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

def find_classes(data: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds class names by scanning the target directory.

    Args:
    - data: pandas DataFrame object containing image paths and corresponding class names

    Returns:
    - Tuple containing:
        - A list of class names
        - A dictionary mapping class names to their index
    """

    classes = ["full", "half", "overflowing"]
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

class CustomImageFolder(Dataset):
    """
    Custom Image folder dataset class.

    Args:
    - data: pandas DataFrame object containing image paths and corresponding class names
    - transform: torchvision.transforms.Compose object containing image transformations

    Attributes:
    - data: pandas DataFrame object containing image paths and corresponding class names
    - transform: torchvision.transforms.Compose object containing image transformations
    - classes: A list of class names
    - class_to_idx: A dictionary mapping class names to their index
    - targets: A list of class labels

    Methods:
    - __len__(self) -> int: Returns the length of the dataset
    - load_image(self, idx: int) -> Image.Image: Loads an image from disk
    - __getitem__(self, idx) -> Tuple[torch.Tensor, int]: Returns the transformed image and its corresponding class label
    """

    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(data)
        self.targets = data.classes.values

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - An integer representing the length of the dataset
        """

        return len(self.data)

    def load_image(self, idx: int) -> Image.Image:
        """
        Loads an image from disk.

        Args:
        - idx: An integer representing the index of the image to load

        Returns:
        - PIL.Image object representing the loaded image
        """

        image_path = self.data.iloc[idx, 0]
        return Image.open(image_path)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Returns the transformed image and its corresponding class label.

        Args:
        - idx: An integer representing the index of the image to load

        Returns:
        - Tuple containing:
            - Transformed image as a torch.Tensor object
            - Corresponding class label as an integer
        """

        img = self.load_image(idx)

        class_name = self.data.iloc[idx, 1]
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx

        else:
            return img, class_idx

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose,batch_size: int) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates and returns train and test dataloaders along with class names.

    Args:
    - train_dir: A string representing the path to the directory containing the training data
    - test_dir: A string representing the path to the directory containing the test data
    - transform: torchvision.transforms.Compose object containing image transformations
    - batch_size: An integer representing the batch size for the dataloaders

    Returns:
    - Tuple containing:
        - Train DataLoader object
        - Test DataLoader object
        - A list of class names
    """
    train_data_transformed = CustomImageFolder(train_dir, transform=transform)
    test_data_transformed = CustomImageFolder(test_dir, transform=transform)

    class_names = train_data_transformed.classes

    train_dataloader = DataLoader(train_data_transformed, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data_transformed, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, class_names
