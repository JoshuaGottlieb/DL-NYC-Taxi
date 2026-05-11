import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple

def load_epoch_data(log_dir: str, model_type: str) -> pd.DataFrame:
    """
    Load and aggregate epoch-level training logs for models of a given type.

    This function scans the specified logging directory for subdirectories containing
    training logs (CSV files) corresponding to a particular model type. It reads and
    concatenates all `training_log.csv` files into a single DataFrame while extracting
    metadata such as optimizer name and learning rate from model directory names.

    Args:
        log_dir (str):
            Path to the directory containing model training logs.
        model_type (str):
            Model type identifier (e.g., "mlp", "dnn", "lr") used to filter directories.

    Returns:
        pd.DataFrame:
            A tidy DataFrame containing model name, optimizer, learning rate,
            and key training metrics across epochs (`epoch`, `mse`, `mae`, `val_mse`, `val_mae`).

    Notes:
        - Assumes each model directory within `log_dir` contains a `training_log.csv` file.
        - Expects directory names to include patterns like `opt_<optimizer>` and `rate_<lr>`.
    """
    # Identify model directories matching the specified type
    model_dirs: List[str] = [
        path for path in os.listdir(log_dir)
        if "model" in path and model_type in path and 'torch' not in path
    ]

    # Initialize an empty DataFrame to collect all epoch logs
    epoch_stats = pd.DataFrame()

    # Iterate through model directories and aggregate their logs
    for path in model_dirs:
        csv_path = os.path.join(log_dir, path, "training_log.csv")

        if not os.path.exists(csv_path):
            continue  # Skip missing log files

        tmp = pd.read_csv(csv_path)
        tmp.insert(0, "model", path)  # Add model name as first column
        epoch_stats = pd.concat([epoch_stats, tmp], axis = 0, ignore_index = True)

    # Extract optimizer name from model folder naming convention
    epoch_stats["optimizer"] = (
        epoch_stats["model"]
        .apply(lambda x: re.search(r"opt_([a-z]+)", x)[1] if re.search(r"opt_([a-z]+)", x) else None)
        .map({"adam": "Adam", "rmsprop": "RMSprop", "sgd": "SGD"})
    )

    # Extract learning rate as float from model folder naming convention
    epoch_stats["learning_rate"] = epoch_stats["model"].apply(
        lambda x: float(re.search(r"rate_([0-9\.]+)", x)[1]) if re.search(r"rate_([0-9\.]+)", x) else None
    )

    # Add RMSE columns
    epoch_stats["rmse"] = np.sqrt(epoch_stats["mse"])
    epoch_stats["val_rmse"] = np.sqrt(epoch_stats["val_mse"])

    # Select relevant columns for final summary
    epoch_stats = epoch_stats[["epoch", "optimizer", "learning_rate", "mse",
                               "rmse", "mae", "val_mse", "val_rmse", "val_mae"]]

    return epoch_stats

def load_model_times(log_dir: str) -> pd.DataFrame:
    """
    Load and summarize model training times and epoch statistics from a log directory.

    This function reads training logs for each model, extracts the number of epochs,
    and merges the data with recorded fit times. It also parses optimizer names,
    learning rates, and computes mean epoch durations.

    Args:
        log_dir (str): Path to the directory containing model subdirectories and 
            a `fit_times.csv` file.

    Returns:
        pd.DataFrame: A DataFrame summarizing model training information with columns:
            ['model', 'optimizer', 'learning_rate', 'epochs', 'mean_epoch_time'].

    Raises:
        FileNotFoundError: If `fit_times.csv` is missing from the specified directory.
        ValueError: If any model directory lacks a valid `training_log.csv` file or
            if model naming patterns are not found.
    """
    # Verify that the fit times file exists
    fit_times_path = os.path.join(log_dir, 'fit_times.csv')
    if not os.path.exists(fit_times_path):
        raise FileNotFoundError(f"Missing required file: {fit_times_path}")

    model_fit_stats = []

    # Collect the number of epochs from each model’s training log
    for model_dir in os.listdir(log_dir):
        full_path = os.path.join(log_dir, model_dir)
        if os.path.isdir(full_path):
            log_path = os.path.join(full_path, 'training_log.csv')
            if not os.path.exists(log_path):
                raise ValueError(f"Missing training log in: {full_path}")
            tmp = pd.read_csv(log_path)
            model_fit_stats.append([model_dir, tmp.shape[0]])

    # Load timing data
    time_stats = pd.read_csv(fit_times_path)

    # Combine epoch and timing data
    model_fit_df = pd.DataFrame(model_fit_stats, columns = ['model_name', 'epochs'])
    model_fit_df = pd.merge(model_fit_df, time_stats, on = 'model_name')

    # Extract model metadata using regex
    model_fit_df['model'] = model_fit_df['model_name'].apply(
        lambda x: re.search(r'model_([a-z]+)', x)[1]
    )
    model_fit_df['optimizer'] = model_fit_df['model_name'].apply(
        lambda x: re.search(r'opt_([a-z]+)', x)[1]
    )
    model_fit_df['learning_rate'] = model_fit_df['model_name'].apply(
        lambda x: float(re.search(r'rate_([0-9\.]+)', x)[1])
    )

    # Compute mean epoch time
    model_fit_df['mean_epoch_time'] = np.round(
        model_fit_df['fit_time'] / model_fit_df['epochs'], 2
    )

    # Reorder columns for clarity
    model_fit_df = model_fit_df[
        ['model', 'optimizer', 'learning_rate', 'epochs', 'mean_epoch_time']
    ]

    return model_fit_df

def load_torch_data(log_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess PyTorch training log and timing data.

    This function reads a model's per-epoch training log (containing loss metrics)
    and total fit time statistics from CSV files located in the specified log directory.
    It computes the RMSE and validation RMSE from loss values, estimates the mean
    epoch duration, and returns both the timing and epoch-level data.

    Args:
        log_dir (str): Directory path containing PyTorch training log files.
            Expected files:
                - '<model_name>/training_log.csv': contains 'train_loss' and 'val_loss'.
                - 'fit_times_torch.csv': contains total 'fit_time' for the model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - torch_times (pd.DataFrame): DataFrame containing platform, number of epochs,
              and mean epoch time for PyTorch models.
            - torch_epoch_data (pd.DataFrame): DataFrame containing per-epoch training
              and validation metrics (loss, RMSE, and validation RMSE).
    """

    # Load per-epoch training and validation losses
    torch_epoch_data = pd.read_csv(
        os.path.join(log_dir, 'model_dnn-opt_sgd-rate_0.001-torch/training_log.csv')
    )

    # Compute RMSE and validation RMSE from MSE losses
    torch_epoch_data['rmse'] = np.sqrt(torch_epoch_data['train_loss'])
    torch_epoch_data['val_rmse'] = np.sqrt(torch_epoch_data['val_loss'])

    # Load total training time data
    torch_times = pd.read_csv(os.path.join(log_dir, 'fit_times_torch.csv'))

    # Add epoch count and compute mean epoch time
    torch_times['epochs'] = torch_epoch_data.shape[0]
    torch_times['mean_epoch_time'] = np.round(torch_times['fit_time'] / torch_times['epochs'], 2)

    # Label and reorder columns for clarity
    torch_times['platform'] = 'PyTorch'
    torch_times = torch_times[['platform', 'epochs', 'mean_epoch_time']]

    return torch_times, torch_epoch_data