"""
Modular Data Setup

This module contains functions and classes for setting up data for machine learning models.

Functions:
- create_dataframe(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]: Creates a Pandas dataframe containing the path of all images and their respective classes for training and testing
- augment_dataset(dataset_dir: str) -> torch.utils.data.Dataset: Loads and augments images from a directory using torchvision's ImageFolder class
- find_classes(data) -> Tuple[List[str], Dict[str, int]]: Finds class names by scanning the target directory
- create_dataloaders(train_dir: str,test_dir: str,transform: transforms.Compose,batch_size: int) -> Tuple[DataLoader, DataLoader, List[str]]: Creates and returns train and test dataloaders along with class names

Classes:
- CustomImageFolder(Dataset): Custom Image folder dataset class

"""
import os
import pandas as pd
from typing import List, Tuple, Dict
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def create_dataframe(data_path):
    """Creates a Pandas dataframe containing the path of all images and their respective classes.

    Args:
        data_path (str): Path to the folder containing the images.

    Returns:
        Pandas dataframe: Dataframe containing image paths and their respective classes.
    """
    image_list = []

    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                image_list.append(
                    {
                        "path": os.path.join(dirpath, filename),
                        "classes": os.path.basename(dirpath),
                    }
                )

    df = pd.DataFrame(image_list)
    df["classes"] = df["classes"].map(
        {
            "Full  Water level": "full",
            "Half water level": "half",
            "Overflowing": "overflowing",
        }
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=42, stratify=df["classes"]
    )

    return train_df, test_df


def augment_dataset(dataset_dir):
    """
    Loads and augments images from a directory using torchvision's ImageFolder class.

    Args:
        dataset_dir (str): Path to directory containing images.

    Returns:
        torch.utils.data.Dataset: A PyTorch dataset object containing the images.
    """
    transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=transforms)

    count = len(dataset)
    if count < 1000:
        while count < 1000:
            for i, (image, label) in enumerate(dataset):
                new_filename = f"image{count + i + 1:03d}.jpeg"
                new_path = os.path.join(dataset_dir, new_filename)
                torchvision.utils.save_image(image, new_path)
                count = len(os.listdir(dataset_dir))
                if count >= 1000:
                    break
            dataset = torchvision.datasets.ImageFolder(
                dataset_dir, transform=transforms
            )

    return dataset


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


def create_dataloaders(
    train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int
) -> Tuple[DataLoader, DataLoader, List[str]]:
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

    train_dataloader = DataLoader(
        train_data_transformed, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data_transformed, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader, class_names
