"""
Contains functions for training and testing a PyTorch model.

Functions:
- train_step(model, dataloader, loss_fn, optimizer, fbeta_score, f1_score, device)
- test_step(model, dataloader, loss_fn, fbeta_score, f1_score, device)
- train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device)
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np

writer = SummaryWriter()


def train_step(model, dataloader, loss_fn, optimizer, fbeta_score, device):
    """
    Trains a PyTorch model for one epoch on the given dataloader.

    Args:
    - model: The PyTorch model to train.
    - dataloader: A PyTorch DataLoader containing the training data.
    - loss_fn: The loss function to use.
    - optimizer: The optimizer to use for training.
    - fbeta_score: The fbeta score function to use.
    - device: The device to use for training.

    Returns:
    - train_loss: The average loss over the entire training set for this epoch.
    - train_fbeta_score: The average fbeta score over the entire training set for this epoch.
    """
    model.train()
    train_loss, train_fbeta_score = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_fbeta_score += fbeta_score(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_fbeta_score /= len(dataloader)

    return train_loss, train_fbeta_score


def test_step(model, dataloader, loss_fn, fbeta_score, device):
    """
    Evaluate a PyTorch model on a test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The test dataset loader.
        loss_fn (callable): The loss function used to calculate the loss.
        fbeta_score (callable): The F-beta score function used to calculate the F-beta score.
        device (str): The device to use for the evaluation (e.g. "cpu", "cuda").

    Returns:
        Tuple[float, float, float]: The average test loss and F-beta score over the test dataset.
    """
    model.eval()
    test_loss, test_fbeta_score = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss
            test_fbeta_score += fbeta_score(y_pred, y)

        test_loss /= len(dataloader)
        test_fbeta_score /= len(dataloader)

    return test_loss, test_fbeta_score


def train(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    epochs,
    device,
    fbeta_score,
    writer,
):
    """
    Train the model for the given number of epochs and evaluate performance on a validation set.

    Args:
    model (nn.Module): The PyTorch model to train.
    train_dataloader (DataLoader): The DataLoader object containing the training dataset.
    test_dataloader (DataLoader): The DataLoader object containing the testing dataset.
    loss_fn (nn.Module): The loss function to use during training.
    optimizer (Optimizer): The optimizer to use during training.
    epochs (int): The number of epochs to train the model for.
    device (str): The device to use for training (e.g. 'cuda' or 'cpu').
    fbeta_score (function): The fbeta score function to use for evaluation.
    writer: A SummaryWriter() instance to log model results to.

    Returns:
    dict: A dictionary containing the following metrics for each epoch:
    - train_loss: the average training loss for the epoch
    - train_fbeta_score: the average F-beta score for the epoch using the training set
    - test_loss: the average testing loss for the epoch
    - test_fbeta_score: the average F-beta score for the epoch using the testing set

    """

    results = {
        "train_loss": [],
        "train_fbeta_score": [],
        "test_loss": [],
        "test_fbeta_score": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_fbeta_score = train_step(
            model, train_dataloader, loss_fn, optimizer, fbeta_score, device
        )
        test_loss, test_fbeta_score = test_step(
            model, test_dataloader, loss_fn, fbeta_score, device
        )
        # scheduler.step()

        print(
            f"Epoch: {epoch+1} \n "
            f"train_loss: {train_loss:.4f} | "
            f"train_f0.5_score: {np.round(train_fbeta_score.item(),2)*100}% | \n "
            f"val_loss: {test_loss:.4f} | "
            f"val_f0.5_score: {np.round(test_fbeta_score.item(),2)*100}% | "
        )
        # --> lr: {scheduler.get_last_lr()[0]}

        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["train_fbeta_score"].append(train_fbeta_score.cpu().detach().numpy())
        results["test_loss"].append(test_loss.cpu().detach().numpy())
        results["test_fbeta_score"].append(test_fbeta_score.cpu().detach().numpy())

        # if epoch % 5 == 0:

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="F-Beta(0.5) Scoree",
                tag_scalar_dict={
                    "train_f0.5": train_fbeta_score,
                    "test_f0.5": test_fbeta_score,
                },
                global_step=epoch,
            )

            writer.add_graph(
                model=model, input_to_model=torch.rand(32, 3, 224, 224).to(device)
            )

            writer.close()
        else:
            pass
    return results
