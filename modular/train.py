"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime
import os

import torch
from torchmetrics import Accuracy, FBetaScore
import data_setup, engine, model_builder, utils
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

# Setting up data directory
data_path = Path("data/")
data_path.mkdir(parents=True, exist_ok=True)

# Setting up arguments parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")
# Adding arguments for number of epochs, batch size and learning rate
parser.add_argument(
    "--num_epochs",
    default=10,
    type=int,
    help="Number of epochs to train the model (default: 10).",
)

parser.add_argument(
    "--batch_size", default=32, type=int, help="Batch size for training (default: 32)."
)

parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="Learning rate for optimizer (default: 1e-4).",
)

parser.add_argument(
    "--data_dir",
    default=data_path,
    type=str,
    help="The path to the directory where the data files are stored.",
)

parser.add_argument(
    "--full_dir",
    default="data/Full  Water level",
    type=str,
    help="The path to the directory where the full water level data files are stored.",
)

parser.add_argument(
    "--half_dir",
    default="data/Half water level",
    type=str,
    help="The path to the directory where the half water level data files are stored.",
)

parser.add_argument(
    "--overflowing_dir",
    default="data/Overflowing",
    type=str,
    help="The path to the directory where the overflowing water level data files are stored.",
)

parser.add_argument(
    "--model_name",
    default="EfficientNet_B0",
    type=str,
    help="This argument specifies the type of model to be used for a certain task, with the default being the EfficientNet_B0 architecture. default (EfficientNet_B0)",
)

parser.add_argument(
    "--loss_function",
    default="CrossEntropyLoss",
    type=str,
    help="Specifies the loss function to be used for training. Default is 'CrossEntropyLoss'",
)

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MODEL_NAME = args.model_name
LOSS_FN = args.loss_function

print("-" * 50 + "\n")
print(f"[INFO] Model: {MODEL_NAME}")
print(f"[INFO] Loss Function: {LOSS_FN}")
print(f"[INFO] Epochs: {NUM_EPOCHS}")
print(f"[INFO] Batch size: {BATCH_SIZE}")
print(f"[INFO] Learning rate: {LEARNING_RATE}")

data_dir = args.data_dir
full_data_dir = args.full_dir
half_data_dir = args.half_dir
overflowing_data_dir = args.overflowing_dir

print("-" * 50 + "\n")
print(f"[INFO] Data dir: {data_dir}")
print(f"[INFO] Full_data dir: {full_data_dir}")
print(f"[INFO] Half_data dir: {half_data_dir}")
print(f"[INFO] Oveflowing_data dir: {overflowing_data_dir}")

# Setup target device
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


# Creating the model architecture
model, data_transform = model_builder.create_model(model_name=MODEL_NAME, num_classes=3)

# Augmenting data from all types of water levels
data_setup.augment_dataset(class_dir=full_data_dir)
data_setup.augment_dataset(class_dir=half_data_dir)
data_setup.augment_dataset(class_dir=overflowing_data_dir)

# Creating train and test dataframes
train_df, test_df = data_setup.create_dataframe(data_path=data_path)

# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_df,
    test_dir=test_df,
    transform=data_transform,
    batch_size=BATCH_SIZE,
)

# Set evaluation metric
# Beta = 0.5
fbeta_score = FBetaScore(task="multiclass", num_classes=len(class_names), beta=0.5).to(
    device
)


def create_writer(
    loss_fn=f"{LOSS_FN}_fn",
    model_name=f"{MODEL_NAME}",
    epoch=f"{NUM_EPOCHS}_epochs",
    lr=f"{LEARNING_RATE}_lr",
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        epoch (str): Epoch number.
        lr (str): Learning rate used in the experiment.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: Instance of a writer saving to log_dir.


    """
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime(
        "%d-%m-%Y,%H-%M-%S"
    )  # returns current date in DD-MM-YYYY,H-M-S format

    log_dir_parts = ["runs", timestamp, model_name, loss_fn, epoch, lr]

    # Create log directory path
    log_dir = os.path.join(*log_dir_parts)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


# Set loss and optimizer
if LOSS_FN == "CrossEntropyLoss":
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Start the timer
    start_time = timer()

    # Starting training
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=cross_entropy_loss,
        epochs=NUM_EPOCHS,
        fbeta_score=fbeta_score,
        device=device,
        writer=create_writer(),
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

elif LOSS_FN == "MultiMarginLoss":
    mm_loss = torch.nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Start the timer
    start_time = timer()

    # Starting training
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=mm_loss,
        epochs=NUM_EPOCHS,
        fbeta_score=fbeta_score,
        device=device,
        writer=create_writer(),
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

else:
    print("Try 'CrossEntropyLoss' or 'BCELoss'")


# Save the model
model_filepath = (
    f"1_{MODEL_NAME}_{NUM_EPOCHS}_epochs_loss_function_{LOSS_FN}_{LEARNING_RATE}_lr.pth"
)
utils.save_model(model=model, target_dir="models", model_name=model_filepath)
print("-" * 50 + "\n")
