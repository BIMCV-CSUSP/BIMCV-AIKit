from os.path import join

from time import strftime, time

from torch import no_grad, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, validation_loader = None, config: dict = {}):
    """
    Basic training function for PyTorch models.

    Args:
        model: PyTorch model, nn.Module
        train_loader: PyTorch DataLoader object for training data
        validation_loader: PyTorch DataLoader object for validation data (Optional)
        config (dict): {
            checkpoint_interval (Optional, int): number of epochs to save a model checkpoint (defaults to None, don't save)
            device (str): device to train the model (defaults to cuda)
            epochs (int): number of epochs to train the model (defaults to 100)
            experiment_name (str): identifier for the saved weight files and tensorboard logs
            loss_function: PyTorch compatible loss function (defaults to torch.nn.CrossEntropyLoss)
            metrics (list): List of Torchmetrics compatible metric functions (defaults to None)
            optimizer: PyTorch compatible optimizer (defaults to torch.optim.Adadelta)
            save_weights_dir (Optional, str): path to the folder where the weights should be stored (defaults to None, don't save)
            tensorboard_writer (Optional, str): path to the folder where the tensorboard record should be created (defaults to None, don't save)
            validation_interval (int): number of epochs to run a validation cycle (defaults to 1, validate in each epoch)
            verbose (bool): whether to print the results to the terminal (defaults to True)
        }
    """


    checkpoint_interval = config["checkpoint_interval"] if "checkpoint_interval" in config else None
    device = config["device"] if "device" in config else "cuda"
    epochs = config["epochs"] if "epochs" in config else 100
    experiment_name = config["experiment_name"] if "experiment_name" in config else f"{model._get_name()}-{strftime('%d-%b-%Y-%H:%M:%S')}"
    loss_function = config["loss_function"] if "loss_function" in config else CrossEntropyLoss()
    metrics = config["metrics"] if "metrics" in config else None
    optimizer = config["optimizer"] if "optimizer" in config else Adadelta(model.parameters())
    save_weights_dir = config["save_weights_dir"] if "save_weights_dir" in config else None
    tensorboard_writer = SummaryWriter(log_dir=join(config["tensorboard_writer_logdir"], experiment_name)) if "tensorboard_writer_logdir" in config else None
    validation_interval = config["validation_interval"] if "validation_interval" in config else 1
    verbose = config["verbose"] if "verbose" in config else True
    
    model = model.to(device)

    train_epoch_len = len(train_loader)
    if validation_loader:
        validation_epoch_len = len(validation_loader)

    best_loss = 1e10
    best_loss_epoch = -1

    for epoch in range(epochs):
        model.train()
        if verbose:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
        start_time = time()
        
        epoch_loss = 0.0
        if metrics:
            metrics_dict = {name: 0.0 for name in list(metrics.keys())}
        
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs).as_tensor()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if metrics:
                for name, metric_fct in metrics.items():
                    metrics_dict[name] = metric_fct(outputs, labels)
    
        epoch_loss /= train_epoch_len
        if tensorboard_writer:
            tensorboard_writer.add_scalar("Loss/train", epoch_loss, epoch)
        if verbose:
            print(f"Epoch {epoch + 1}: \n\tTrain Loss: {epoch_loss:.4f}")
        if metrics:
            for name, metric_fct in metrics.items():
                value = metric_fct.compute()
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(f"{name}/train", value, epoch)
                if verbose:
                    print(f"\tTrain {name}: {value:.4f}")
                metric_fct.reset()
        
    
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            epoch_loss = 0.0

            for val_data in validation_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                with no_grad():
                    val_outputs = model(val_images).as_tensor()
                    loss = loss_function(val_outputs, val_labels)
                    epoch_loss += loss.item()
                    if metrics:
                        for name, metric_fct in metrics.items():
                            metrics_dict[name] = metric_fct(outputs, labels)
            
                epoch_loss /= validation_epoch_len
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/validation", epoch_loss, epoch)
                if verbose:
                    print(f"\n\tValidation Loss: {epoch_loss:.4f}")
                if metrics:
                    for name, metric_fct in metrics.items():
                        value = metric_fct.compute()
                        if tensorboard_writer:
                            tensorboard_writer.add_scalar(f"{name}/validation", value, epoch)
                        if verbose:
                            print(f"\tValidation {name}: {value:.4f}")
                        metric_fct.reset()
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_epoch = epoch + 1
        
                if save_weights_dir:
                    save(model.state_dict(), join(save_weights_dir, f"{experiment_name}.pth"))
                        
                if verbose:
                    print(f"Lower loss: {best_loss:.4f} at epoch {best_loss_epoch}. Saved new best metric model.")

        epoch_elapsed_time = time()-start_time
        if verbose:
            print(f"Epoch elapsed time: {epoch_elapsed_time:.4f}")
        if tensorboard_writer:
            tensorboard_writer.add_scalar("epoch_elapsed_time", epoch_elapsed_time, epoch)
        
        if checkpoint_interval:
            if save_weights_dir and (epoch + 1) % checkpoint_interval == 0:
                save(model.state_dict(), join(save_weights_dir, f"{experiment_name}_{epoch+1}.pth"))
                if verbose:
                    print(f"Saved checkpoint at epoch {epoch+1}")

    if verbose:
        print(f"Training completed, lowest validation loss value: {best_loss:.4f} at epoch: {best_loss_epoch}.")