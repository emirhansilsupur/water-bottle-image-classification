"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
from pathlib import Path
import torch
from torchmetrics import F1Score, FBetaScore
from modular import data_setup, engine, model_builder, utils

# Setting up data directory
data_path = (Path("data/"),)
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
    default="data\Full  Water level",
    type=str,
    help="The path to the directory where the full water level data files are stored.",
)

parser.add_argument(
    "--half_dir",
    default="data\Half water level",
    type=str,
    help="The path to the directory where the half water level data files are stored.",
)

parser.add_argument(
    "--overflowing_dir",
    default="data\Overflowing",
    type=str,
    help="The path to the directory where the overflowing water level data files are stored.",
)

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

print(f"[INFO] Number of epochs: {NUM_EPOCHS}")
print(f"[INFO] Batch size: {BATCH_SIZE}")
print(f"[INFO] Learning rate: {LEARNING_RATE}")

data_dir = args.data_dir
full_data_dir = args.full_dir
half_data_dir = args.half_dir
overflowing_data_dir = args.overflowing_dir

print(f"[INFO] Data dir: {data_dir}")
print(f"[INFO] Full_data dir: {full_data_dir}")
print(f"[INFO] Half_data dir: {half_data_dir}")
print(f"[INFO] Oveflowing_data dir: {overflowing_data_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating the model using EfficientNet_B0 architecture
model_name = "EfficientNet_B0"
model, data_transform = model_builder.create_model(model_name=model_name, num_classes=3)

# Augmenting data from all types of water levels
augmented_full_data = data_setup.augment_dataset(
    class_dir=full_data_dir, dataset_dir=data_dir
)
augmented_half_data = data_setup.augment_dataset(
    class_dir=half_data_dir, dataset_dir=data_dir
)
augmented_overflowing_data = data_setup.augment_dataset(
    class_dir=overflowing_data_dir, dataset_dir=data_dir
)
# Merging all the augmented data
augmented_data = [augmented_full_data, augmented_half_data, augmented_overflowing_data]

# Creating train and test dataframes
train_df, test_df = data_setup.create_dataframe(augmented_data)

# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_df,
    test_dir=test_df,
    transform=data_transform,
    batch_size=BATCH_SIZE,
)
# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Set evaluation metrics
fbeta_score = FBetaScore(task="multiclass", num_classes=3, beta=0.5).to(device)
f1_score = F1Score(task="multiclass", num_classes=3, average="macro").to(device)

# Starting training
results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    fbeta_score=fbeta_score,
    f1_score=f1_score,
    device=device,
)

# Save the model
utils.save_model(model=model, target_dir="models", model_name="model_name.pth")
