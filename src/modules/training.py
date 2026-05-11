import os
import time
from timeit import default_timer as timer
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from keras import Model
from keras.callbacks import Callback
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, History, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Optimizer, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

# ---- TENSORFLOW FUNCTIONS ----
# ---- Neural Net Models ----

def multilayer_perceptron(input_size: int, optimizer: Optimizer,
                          num_layers: int = 2, loss: str = 'mse') -> Model:
    """
    Build and compile a multilayer perceptron (MLP) model for regression tasks.

    This function constructs a fully connected neural network using Keras,
    where each hidden layer applies a LeakyReLU activation to mitigate the
    "dying ReLU" problem. The model is intended for continuous output regression,
    such as predicting taxi trip durations, fares, or other numerical targets.

    Args:
        input_size (int):
            Number of input features in the dataset.
        optimizer (Optimizer):
            A Keras optimizer instance (e.g., Adam, RMSprop, SGD) used for model training.
        num_layers (int, optional):
            Number of hidden layers to include in the network. Defaults to 2.
        loss (str, optional):
            Loss function to optimize during training (e.g., 'mse', 'mae'). Defaults to 'mse'.

    Returns:
        Model:
            A compiled Keras `Model` object ready for training and evaluation.

    Notes:
        - Each hidden layer uses the same number of units as the input dimension.
        - LeakyReLU activation is used to prevent inactive neurons.
        - The output layer uses ReLU activation for non-negative regression outputs.
        - Default metrics include Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    """

    # Step 1: Define the input layer based on the feature dimension
    inputs = Input(shape = (input_size,))
    x = inputs

    # Step 2: Construct hidden layers dynamically using LeakyReLU activation
    for i in range(num_layers):
        dense = Dense(input_size, activation = LeakyReLU(negative_slope = 0.3))
        x = dense(x)

    # Step 3: Add the output layer with ReLU activation for non-negative predictions
    outputs = Dense(1, activation = 'relu')(x)

    # Step 4: Build the MLP model
    model = Model(inputs = inputs, outputs = outputs, name = f"mlp-{num_layers}")

    # Step 5: Compile the model with the specified optimizer, loss, and metrics
    model.compile(loss = loss, optimizer = optimizer, metrics = ['mse', 'mae'])

    # Step 6: Return the compiled Keras model
    return model

def linear_regression(input_size: int, optimizer: Optimizer, loss: str = 'mse') -> Model:
    """
    Build and compile a simple linear regression model using Keras.

    This function constructs a single-layer neural network (no hidden layers)
    that performs linear regression on input features. The model applies a ReLU
    activation on the output to enforce non-negative predictions, which is often
    suitable for continuous target variables like fares, durations, or other
    positive quantities.

    Args:
        input_size (int):
            Number of input features in the dataset.
        optimizer (Optimizer):
            A Keras optimizer instance (e.g., Adam, RMSprop, SGD) used for model training.
        loss (str, optional):
            Loss function to minimize during training (e.g., 'mse', 'mae'). Defaults to 'mse'.

    Returns:
        Model:
            A compiled Keras `Model` object representing a linear regression network.

    Notes:
        - The model has no hidden layers, representing a pure linear mapping
          from inputs to a single output neuron.
        - The output layer uses ReLU activation to ensure non-negative predictions.
        - Default evaluation metrics include Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    """

    # Step 1: Define the input layer with shape equal to the number of features
    inputs = Input(shape = (input_size,))
    x = inputs

    # Step 2: Add a single output neuron with ReLU activation
    # This performs a linear transformation followed by non-negativity constraint
    outputs = Dense(1, activation = 'relu')(x)

    # Step 3: Build the Keras Model
    model = Model(inputs = inputs, outputs = outputs, name = 'linear_regression')

    # Step 4: Compile the model with the specified optimizer, loss, and metrics
    model.compile(loss = loss, optimizer = optimizer, metrics = ['mse', 'mae'])

    # Step 5: Return the compiled linear regression model
    return model

def feed_forward_dnn(input_size: int, optimizer: Optimizer,
                     num_layers: int = 4, loss: str = 'mse') -> Model:
    """
    Build and compile a deep feed-forward neural network (DNN) with layer-wise 
    shrinking width and regularization for regression tasks.

    This model progressively decreases the number of neurons per layer, applies 
    LeakyReLU activations for non-linearity, L2 regularization to prevent overfitting, 
    and dropout for additional regularization. The network outputs a single 
    non-negative regression prediction using a ReLU activation in the final layer.

    Args:
        input_size (int):
            Number of input features in the dataset.
        optimizer (Optimizer):
            A Keras optimizer instance (e.g., Adam, RMSprop, SGD) used for model training.
        num_layers (int, optional):
            Number of fully connected (Dense) layers in the network. Defaults to 4.
        loss (str, optional):
            Loss function to minimize during training (e.g., 'mse', 'mae'). Defaults to 'mse'.

    Returns:
        Model:
            A compiled Keras `Model` object representing the deep feed-forward neural network.

    Notes:
        - Each Dense layer’s width decreases exponentially max(64 / (2 ** i), 16).
        - LeakyReLU (negative_slope = 0.3) prevents dying ReLU issues in deep layers.
        - Dropout (rate = 0.25) is applied to early layers to improve generalization.
        - L2 regularization is used on both kernel and bias to reduce overfitting.
        - Output activation is ReLU to constrain predictions to non-negative values.
    """

    # Step 1: Define the input layer based on the number of features
    inputs = Input(shape = (input_size,))
    x = inputs

    # Step 2: Build a series of Dense layers with decreasing width
    for i in range(num_layers):
        # Each layer width halves until reaching a minimum of 16 neurons or 1024 max
        dense = Dense(
            max(64 // (2 ** i), 16),
            activation = LeakyReLU(negative_slope = 0.3), # Nonlinear activation to prevent dead neurons
            kernel_regularizer = l2(0.001), # L2 penalty on weights
            bias_regularizer = l2(0.001) # L2 penalty on biases
        )
        x = dense(x)

        # Apply dropout in the first half of layers to improve generalization
        if i <= (num_layers - 1) // 2:
            dropout = Dropout(0.25)
            x = dropout(x)

    # Step 3: Output layer for regression — single neuron with ReLU activation
    outputs = Dense(1, activation = 'relu')(x)

    # Step 4: Build and compile the model
    model = Model(inputs = inputs, outputs = outputs, name = f'ffdnn-{num_layers}')
    model.compile(loss = loss, optimizer = optimizer, metrics = ['mse', 'mae'])

    # Step 5: Return the compiled DNN model
    return model

# ---- Model Setup Functions ----

def set_learning_rate(initial_lr: float, decay_steps: int) -> ExponentialDecay:
    """
    Create an exponentially decaying learning rate schedule for Keras optimizers.

    This function returns a learning rate schedule that decreases the learning rate 
    exponentially at discrete intervals (steps), which can help stabilize training 
    and prevent overshooting the optimal minimum.

    Args:
        initial_lr (float):
            Initial learning rate at the start of training.
        decay_steps (int):
            Number of training steps (batches) after which the learning rate is decayed.

    Returns:
        ExponentialDecay:
            A TensorFlow Keras learning rate schedule object, compatible with any optimizer.

    Notes:
        - The decay rate is fixed at 0.96 per `decay_steps`, leading to
            0.01687 the initial learning rate after 100 epochs.
        - The schedule uses `staircase = True`, meaning the learning rate decreases 
          in discrete steps rather than continuously.
        - The effective learning rate is computed as:
              lr(step) = initial_lr * decay_rate^(floor(step / decay_steps))
        - Commonly used with optimizers such as Adam, RMSprop, or SGD.
    """
    return ExponentialDecay(
        initial_learning_rate = initial_lr,
        decay_steps = decay_steps,
        decay_rate = 0.96,
        staircase = True
    )

def select_optimizer(optimizer_name: str, lr: Optional[Union[float, int]] = None) -> Optimizer:
    """
    Select and initialize a Keras optimizer based on the provided name and learning rate.

    This function simplifies the creation of commonly used optimizers by 
    automatically assigning a default learning rate if none is specified.
    Supported optimizers:
      - 'SGD': Stochastic Gradient Descent
      - 'RMSprop': Root Mean Square Propagation
      - 'Adam': Adaptive Moment Estimation

    Args:
        optimizer_name (str):
            Name of the optimizer to instantiate. Must be one of {'SGD', 'RMSprop', 'Adam'}.
        lr (Optional[Union[float, int]], optional):
            Custom learning rate for the optimizer. 
            If None, a default is chosen:
              * SGD → 0.01  
              * RMSprop → 0.001  
              * Adam → 0.001  

    Returns:
        Optimizer:
            A compiled Keras optimizer instance corresponding to the requested optimizer.

    Raises:
        ValueError: If `optimizer_name` is not one of 'SGD', 'RMSprop', or 'Adam'.

    Notes:
        - Default learning rates follow Keras best practices.
        - This function is useful for dynamically selecting optimizers in experiments or pipelines.
    """

    if optimizer_name == 'SGD':
        return SGD(learning_rate = lr if lr is not None else 0.01)
    elif optimizer_name == 'RMSprop':
        return RMSprop(learning_rate = lr if lr is not None else 0.001)
    elif optimizer_name == 'Adam':
        return Adam(learning_rate = lr if lr is not None else 0.001)

    # Raise an error for invalid optimizer names
    raise ValueError("Optimizer must be one of 'SGD', 'RMSprop', or 'Adam'.")

class TimingCallback(Callback):
    """
    Keras callback to measure and log the duration of each training epoch.

    This callback records the time taken for each epoch during model training
    and appends the total training time to a CSV file after training completes.

    Args:
        model_name (str):
            Name of the model, used for logging purposes.
        path (str):
            File path to a CSV file where the total training time will be appended.

    Attributes:
        logs (list[float]):
            List storing the duration (in seconds) of each epoch.
        model_name (str):
            Name of the model being trained.
        path (str):
            Path to the CSV file for logging total training time.
    """

    def __init__(self, model_name: str, path: str):
        # Initialize attributes
        super().__init__()
        self.logs: list[float] = []  # Stores epoch durations
        self.model_name: str = model_name # Model name for logging
        self.path: str = path # CSV file path

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the beginning of each epoch to start the timer.

        Args:
            epoch (int): Current epoch index.
            logs (dict, optional): Metric results for this epoch (ignored here).
        """
        self.starttime: float = timer()  # Record start time for the epoch

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the end of each epoch to record elapsed time.

        Args:
            epoch (int): Current epoch index.
            logs (dict, optional): Metric results for this epoch (ignored here).
        """
        epoch_time: float = timer() - self.starttime
        self.logs.append(epoch_time)  # Save duration of this epoch

    def on_train_end(self, logs: dict = None) -> None:
        """
        Called at the end of training to compute total time and save to CSV.

        Args:
            logs (dict, optional): Final metric results (ignored here).
        """
        # Compute total training time by summing all epoch durations
        training_time: float = float(np.sum(self.logs))

        # Create a DataFrame with model name and total training time
        df = pd.DataFrame(
            [[self.model_name, training_time]],
            columns = ['model_name', 'time']
        )

        # Append total time to the CSV file without writing headers or index
        df.to_csv(self.path, mode = 'a', header = False, index = False)

def set_callbacks(model_name: str, model_type: str) -> List[Callback]:
    """
    Configure and return a list of Keras callbacks for model training.

    This function prepares a set of standard callbacks to monitor and log model training,
    save checkpoints, record training times, and implement early stopping. Each callback
    contributes to better tracking, reproducibility, and efficient model management.

    Args:
        model_name (str):
            Name of the model instance, used to define file paths for logs and timing outputs.
        model_type (str):
            Model category (e.g., "mlp", "dnn", "lr") used to organize model checkpoint directories.

    Returns:
        List[Callback]:
            A list of initialized Keras callbacks for use with `model.fit()`.

    Callbacks:
        - **ModelCheckpoint**: Saves model weights after each epoch.
        - **TensorBoard**: Logs training metrics for visualization.
        - **CSVLogger**: Records training history to a CSV file.
        - **TimingCallback**: Tracks and logs total training time.
        - **EarlyStopping**: Stops training early if validation loss does not improve.
    """

    # Callback to save model weights after each epoch
    ckpt = ModelCheckpoint(
        filepath = f'../models/{model_type}/{model_name}-{{epoch:03d}}.weights.h5', # Filepath with epoch number
        monitor = 'val_loss', # Metric to monitor
        save_best_only = False, # Save weights every epoch, not just the best
        save_weights_only = True, # Save only weights, not full model
        save_freq = 'epoch' # Save after each epoch
    )

    # Callback for TensorBoard visualization
    board = TensorBoard(
        log_dir = f'../logs/{model_name}' # Directory to store logs
    )

    # Callback to log training metrics to CSV
    logger = CSVLogger(
        filename = f'../logs/{model_name}/training_log.csv' # CSV log file path
    )

    # Custom callback to measure training time per epoch and total
    timer = TimingCallback(
        model_name = model_name,
        path = '../logs/fit_times.csv' # File to append total training time
    )

    stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 8,
        restore_best_weights = True
    )

    # Return all callbacks as a list
    return [ckpt, board, logger, timer, stopping]

# ---- Model Training ----

def fit_tf_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int, model_type: str, model_name: str, initial_lr: float,
                 optimizer: str, loss: str = 'mse') -> History:
    """
    Compile and train a Keras model with specified architecture, optimizer, 
    learning rate schedule, and callbacks.

    This function supports three model types:
      - 'mlp': Shallow multilayer perceptron
      - 'lr': Single-layer linear regression
      - 'dnn': Deep feed-forward neural network

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target vector.
        epochs (int): Number of training epochs.
        model_type (str): Type of model to train ('mlp', 'lr', or 'dnn').
        model_name (str): Name of the model (used for logging and saving).
        initial_lr (float): Initial learning rate for the optimizer.
        optimizer (str): Optimizer name ('SGD', 'RMSprop', or 'Adam').
        loss (str, optional): Loss function to optimize ('mse', 'mae', etc.). Defaults to 'mse'.

    Returns:
        History: Keras History object containing training metrics for each epoch.

    Notes:
        - Sets up callbacks for checkpointing, TensorBoard logging, CSV logging, and timing.
        - Applies an exponential learning rate decay schedule via `set_learning_rate`.
        - Selects and configures the optimizer using `select_optimizer`.
        - Batch size is fixed at 32 for all model types.
    """

    # Step 1: Setup callbacks for logging, checkpointing, and timing
    callbacks = set_callbacks(model_name, model_type)

    # Step 2: Configure exponential learning rate schedule
    decay_steps = X_train.shape[0] // 32
    lr_scheduler = set_learning_rate(initial_lr, decay_steps = decay_steps)

    # Step 3: Select optimizer with the learning rate schedule
    opt = select_optimizer(optimizer, lr_scheduler)

    # Step 4: Determine input size from training data
    input_size = X_train.shape[1]

    # Step 5: Instantiate the model based on the specified type
    if model_type == 'mlp':
        model = multilayer_perceptron(input_size = input_size, optimizer = opt, loss = loss, num_layers = 2)
    elif model_type == 'lr':
        model = linear_regression(input_size = input_size, optimizer = opt, loss = loss)
    elif model_type == 'dnn':
        model = feed_forward_dnn(input_size = input_size, optimizer = opt, loss = loss, num_layers = 4)
    else:
        raise ValueError("Invalid model_type. Must be one of 'mlp', 'lr', or 'dnn'.")

    # Step 6: Convert NumPy arrays to TensorFlow Datasets
    batch_size = 32
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size = len(X_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Step 7: Train the model
    history = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs = epochs,
        callbacks = callbacks,
        verbose = 1
    )
    model.save(f'../models/{model_type}/{model_name}-best_model.keras')

    # Step 8: Return training history
    return history

# ---- PYTORCH FUNCTIONS ----
# ---- Model Class ----

class FeedForwardDNN(nn.Module):
    """
    Deep feed-forward neural network (DNN) for regression tasks.

    The model architecture consists of progressively shrinking fully connected 
    layers with LeakyReLU activations, dropout regularization in the earlier layers, 
    and an output ReLU to enforce non-negative predictions. L2 regularization is 
    intended to be applied externally through the optimizer's `weight_decay` parameter.

    Args:
        input_size (int): Number of input features.
        num_layers (int, optional): Number of hidden layers in the network. 
            Defaults to 4. The size of each layer decreases by half until 
            reaching a minimum of 16 units.

    Attributes:
        model (nn.Sequential): The sequential neural network architecture 
            containing linear, activation, and dropout layers.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size).

    Forward Output:
        torch.Tensor: Model output tensor of shape (batch_size, 1), representing 
            non-negative regression predictions.
    """

    def __init__(self, input_size: int, num_layers: int = 4) -> None:
        super().__init__()

        layers = []
        in_features = input_size

        # Construct progressively smaller hidden layers
        for i in range(num_layers):
            out_features = max(64 // (2 ** i), 16)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(negative_slope = 0.3))

            # Apply dropout to early layers to prevent overfitting
            if i <= (num_layers - 1) // 2:
                layers.append(nn.Dropout(0.25))

            in_features = out_features

        # Final output layer for regression (ReLU ensures non-negative predictions)
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.ReLU())

        # Combine all layers into a single sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing 
            non-negative regression predictions.
        """
        return self.model(x)

# ---- Optimizer and Learning Rate -----

def select_optimizer_torch(model: nn.Module, optimizer_name: str,
                           lr: Optional[float] = None, weight_decay: float = 0.001) -> Optimizer:
    """
    Select and initialize a PyTorch optimizer with configurable learning rate 
    and optional L2 regularization (via weight decay).

    This function provides a unified interface to initialize one of the three
    most commonly used optimizers in PyTorch — SGD, RMSprop, or Adam — while
    applying consistent learning rate and weight decay defaults.

    Args:
        model (nn.Module):
            The PyTorch model whose parameters will be optimized.
        optimizer_name (str):
            Name of the optimizer to use. Must be one of:
            `'SGD'`, `'RMSprop'`, or `'Adam'`.
        lr (Optional[float], optional):
            Learning rate for the optimizer. If not provided, a default is chosen 
            based on the optimizer type:
                - SGD: 0.01  
                - RMSprop: 0.001  
                - Adam: 0.001
        weight_decay (float, optional):
            L2 regularization factor (default = 0.001).

    Returns:
        Optimizer:
            An initialized PyTorch optimizer ready for training.

    Raises:
        ValueError:
            If `optimizer_name` is not one of `'SGD'`, `'RMSprop'`, or `'Adam'`.
    """

    # Initialize the selected optimizer
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr = lr if lr else 0.01, weight_decay = weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr = lr if lr else 0.001, weight_decay = weight_decay)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr = lr if lr else 0.001, weight_decay = weight_decay)
    else:
        raise ValueError("Optimizer must be one of 'SGD', 'RMSprop', or 'Adam'.")

def set_learning_rate_torch(optimizer: Optimizer, decay_steps: int, decay_rate: float = 0.96) -> StepLR:
    """
    Create an exponential (step-based) learning rate scheduler for a PyTorch optimizer.

    This scheduler reduces the learning rate by a constant factor (`decay_rate`)
    every `decay_steps` epochs. Gradual learning rate decay can improve model
    convergence stability and help prevent overshooting during later stages of training.

    Args:
        optimizer (Optimizer):
            A PyTorch optimizer whose learning rate will be scheduled.
        decay_steps (int):
            The number of epochs between each learning rate decay step.
        decay_rate (float, optional):
            The multiplicative decay factor applied to the learning rate 
            after every `decay_steps` epochs. Defaults to 0.96.

    Returns:
        StepLR:
            A PyTorch learning rate scheduler object that can be stepped 
            each epoch (via `scheduler.step()`).

    Notes:
        - The learning rate at step `t` is updated as:
              lr_t = lr_0 * (decay_rate)^(floor(t / decay_steps))
        - This scheduler applies a piecewise constant decay rather than a smooth exponential curve.
    """

    # Initialize an exponential-style step learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer = optimizer,
        step_size = decay_steps, # Number of epochs between decay events
        gamma = decay_rate # Multiplicative factor for LR decay
    )

    # Return the scheduler object for use during training
    return scheduler

# ---- Loggers (equivalent to Callbacks in TF) ----

class TorchTimingLogger:
    """
    Utility class to log per-epoch training durations and record total training time to a CSV file.

    This lightweight logger is designed for use with custom PyTorch training loops.
    It measures the duration of each training epoch and, upon completion, writes the
    total training time for the model to a specified CSV file.

    Args:
        model_name (str):
            Name of the model being trained, recorded in the log file.
        path (str):
            File path to append total training time data (e.g., '../logs/fit_times.csv').

    Attributes:
        model_name (str):
            The name of the model being logged.
        path (str):
            The output CSV file path for saving timing data.
        epoch_times (List[float]):
            A list storing epoch durations in seconds.
    """

    def __init__(self, model_name: str, path: str) -> None:
        self.model_name = model_name
        self.path = path
        self.epoch_times = [] # Store durations of each epoch

    def on_epoch_begin(self) -> None:
        """Mark the start time of an epoch using a high-resolution performance counter."""
        self.start_time = time.perf_counter()

    def on_epoch_end(self) -> None:
        """Record the elapsed time for the completed epoch."""
        self.epoch_times.append(time.perf_counter() - self.start_time)

    def on_train_end(self) -> None:
        """
        Compute and save the total training time to a CSV file.

        Appends a row containing the model name and total elapsed time (in seconds)
        to the specified CSV file. Creates the directory if it does not exist.
        """
        # Compute total training time across all epochs
        total_time = float(np.sum(self.epoch_times))

        # Create a DataFrame with model name and total time
        df = pd.DataFrame([[self.model_name, total_time]], columns = ["model_name", "time"])

        # Ensure log directory exists before writing
        os.makedirs(os.path.dirname(self.path), exist_ok = True)

        # Append results to CSV file (no header after first write)
        df.to_csv(self.path, mode = "a", header = False, index = False)

class TorchEarlyStopping:
    """
    Implements early stopping for PyTorch training loops, similar to Keras' EarlyStopping.

    Monitors validation loss and stops training if no improvement is observed
    for a specified number of consecutive epochs (`patience`). Optionally restores
    the model weights corresponding to the best validation loss.

    Args:
        patience (int, optional):
            Number of epochs with no improvement after which training will be stopped.
            Defaults to 8.
        restore_best_weights (bool, optional):
            If True, restores the model parameters to the state that achieved
            the best validation loss when stopping. Defaults to True.

    Attributes:
        best_loss (float):
            Lowest validation loss observed so far.
        counter (int):
            Number of consecutive epochs without improvement.
        best_state (Optional[dict]):
            State dictionary of the model corresponding to `best_loss`.
        should_stop (bool):
            Flag indicating whether training should stop.
    """

    def __init__(self, patience: int = 8, restore_best_weights: bool = True) -> None:
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """
        Evaluate whether to stop training early based on validation loss.

        If the validation loss improves, resets the counter and optionally
        saves the model state. If no improvement occurs for `patience` epochs,
        sets `should_stop` to True.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (nn.Module): PyTorch model being trained.
        """
        if val_loss < self.best_loss:
            # Validation loss improved: reset counter and save best model state
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            # No improvement: increment counter
            self.counter += 1
            if self.counter >= self.patience:
                # Patience exceeded: mark for early stopping
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        """
        Restore model weights to the state corresponding to the best validation loss.

        Args:
            model (nn.Module): PyTorch model to restore.
        """
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)

class TorchCSVLogger:
    """
    Logs training metrics per epoch to a CSV file, similar to Keras' CSVLogger.

    Records metrics such as training and validation loss for each epoch
    and appends them to a CSV file. Creates the directory if it does not exist
    and writes headers only once.

    Args:
        filename (str):
            Path to the CSV file where metrics will be saved. If the directory
            does not exist, it will be created automatically.

    Attributes:
        filename (str): Path to the CSV file.
        columns_written (bool): Tracks whether CSV header has already been written.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        # Tracks if header has been written to CSV
        self.columns_written = False

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Log metrics for a single epoch to the CSV file.

        Appends a new row with epoch number, training loss, and validation loss.
        Writes CSV headers only once on the first write.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
        """
        # Create a one-row DataFrame for current epoch metrics
        df = pd.DataFrame([[epoch, train_loss, val_loss]],
                          columns = ['epoch', 'train_loss', 'val_loss'])
        # Append to CSV file, write header only if not already written
        df.to_csv(self.filename, mode = 'a', header = not self.columns_written, index = False)
        self.columns_written = True

# ---- Training Function ----

def fit_torch_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int, model_type: str, model_name: str, initial_lr: float,
                    optimizer_name: str, loss: str = 'mse', batch_size: int = 32) -> Dict[str, List[float]]:
    """
    Train a PyTorch feed-forward DNN for regression with utilities similar to Keras.

    Includes:
      - Learning rate scheduling
      - Early stopping with optional best-model restoration
      - CSV logging of per-epoch metrics
      - Timing of epoch and total training
      - Saving model weights per epoch and final/best model

    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training target vector of shape (n_samples,).
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target vector.
        epochs (int): Maximum number of training epochs.
        model_type (str): Model type name, used for file paths.
        model_name (str): Unique model identifier, used for logging and saving.
        initial_lr (float): Initial learning rate for the optimizer.
        optimizer_name (str): Name of optimizer: 'SGD', 'RMSprop', or 'Adam'.
        loss (str, optional): Loss function, 'mse' or 'mae'. Defaults to 'mse'.
        batch_size (int, optional): Mini-batch size for training. Defaults to 32.

    Returns:
        Dict[str, List[float]]: Dictionary with keys 'train_loss' and 'val_loss',
        storing per-epoch metrics.
    """

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype = torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype = torch.float32).view(-1, 1)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                            batch_size = batch_size, shuffle = False)

    # Initialize model, optimizer, and scheduler
    input_size = X_train.shape[1]
    model = FeedForwardDNN(input_size = input_size, num_layers = 4).to(device)
    optimizer = select_optimizer_torch(model, optimizer_name, lr = initial_lr)
    scheduler = set_learning_rate_torch(optimizer, decay_steps = 1)

    # Select loss function
    criterion = nn.MSELoss() if loss == 'mse' else nn.L1Loss()

    # Initialize utility callbacks for logging, timing, and early stopping
    timer = TorchTimingLogger(model_name, '../logs/fit_times_torch.csv')
    early_stopping = TorchEarlyStopping(patience = 8, restore_best_weights = True)
    csv_logger = TorchCSVLogger(f'../logs/{model_name}/training_log.csv')

    history = {'train_loss': [], 'val_loss': []}

    # Main training loop
    for epoch in range(epochs):
        timer.on_epoch_begin()
        model.train()
        train_loss = 0.0

        # Training over mini-batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss_value = criterion(outputs, y_batch)
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item() * X_batch.size(0)

        # Update scheduler and average train loss
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        csv_logger.log_epoch(epoch + 1, train_loss, val_loss)
        timer.on_epoch_end()

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save model weights per epoch
        epoch_path = f'../models/{model_type}/{model_name}-{epoch + 1:03d}.pt'
        os.makedirs(os.path.dirname(epoch_path), exist_ok = True)
        torch.save(model.state_dict(), epoch_path)

        # Early stopping check
        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            early_stopping.restore_best(model)
            break

    # Record total training time
    timer.on_train_end()

    # Save final/best model
    save_path = f'../models/{model_type}/{model_name}-best_model.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    torch.save(model.state_dict(), save_path)

    return history