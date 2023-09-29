from os.path import join
from time import sleep, strftime, time
from tqdm import tqdm

from torch import load, no_grad, save, tensor, vstack
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adadelta
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, weights, data_loader, metrics, device: str = "cuda"):
    """
    Basic evaluation function for PyTorch models.

    Args:
        model: PyTorch model, nn.Module
        data_loader: PyTorch DataLoader object
        device (str): device to train the model (defaults to cuda)
        metrics (dict): dict of Torchmetrics compatible metric functions (defaults to None)
        }
    """

    def compute_metrics(predictions, labels):
        if not metrics:
            return
        metrics_dict = {}
        if any(predictions.sum(dim=1) != 1.0):
            predictions = softmax(predictions, dim=1)
        predictions = predictions.argmax(dim=1).to('cpu')
        labels = labels.argmax(dim=1).to('cpu')
        for name, metric_fct in metrics.items():
            if "matrix" not in name:
                metrics_dict[name] = metric_fct(predictions, labels).item()
            else:
                print("\nConfusion Matrix")
                print(metric_fct(predictions, labels).numpy())
        print(f"\nResults: \n{', '.join([name+':'+f'{value:.4f}' for name, value in metrics_dict.items()])}")
        return predictions, labels
    
    model.load_state_dict(load(weights))
    print("Loaded weights "+weights)
    model = model.to(device)
    model.eval()

    outputs = []
    labels = []

    with tqdm(data_loader, unit="batch") as tepoch:
        for batch_data in tepoch:
            tepoch.set_description(f"Progress")
            batch_images, batch_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            with no_grad():
                labels.append(batch_labels)
                outputs.append(model(batch_images))
    predictions, labels = compute_metrics(vstack(outputs), vstack(labels))
    print("\n Predictions: " + str(predictions.numpy()))
    print("True labels: " + str(labels.numpy()))
                

def train(model, train_loader, validation_loader=None, config: dict = {}):
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
            metrics (dict): dict of Torchmetrics compatible metric functions (defaults to None)
            optimizer: PyTorch compatible optimizer (defaults to torch.optim.Adadelta)
            save_weights_dir (Optional, str): path to the folder where the weights should be stored (defaults to None, don't save)
            tensorboard_writer_logdir (Optional, str): path to the folder where the tensorboard record should be created (defaults to None, don't save)
            validation_interval (int): number of epochs to run a validation cycle (defaults to 1, validate in each epoch)
            verbose (bool): whether to print the results to the terminal (defaults to True)
        }
    """

    checkpoint_interval = config["checkpoint_interval"] if "checkpoint_interval" in config else None
    device = config["device"] if "device" in config else "cuda"
    epochs = config["epochs"] if "epochs" in config else 100
    experiment_name = config["experiment_name"] if "experiment_name" in config else f"{model._get_name()}_{strftime('%d-%b-%Y-%H:%M:%S')}"
    loss_function = config["loss_function"] if "loss_function" in config else CrossEntropyLoss()
    metrics = config["metrics"] if "metrics" in config else None
    optimizer = config["optimizer"] if "optimizer" in config else Adadelta(model.parameters())
    save_weights_dir = config["save_weights_dir"] if "save_weights_dir" in config else None
    tensorboard_writer = (
        SummaryWriter(log_dir=join(config["tensorboard_writer_logdir"], experiment_name)) if "tensorboard_writer_logdir" in config else None
    )
    validation_interval = config["validation_interval"] if "validation_interval" in config else 1
    verbose = config["verbose"] if "verbose" in config else True

    def aggregate_metrics_per_epoch(stage):
        if not metrics:
            tepoch.set_postfix(loss=epoch_loss)
            return
        metrics_dict = {}
        for name, metric_fct in metrics.items():
            metrics_dict[name] = f"{metric_fct.compute()}"
            if tensorboard_writer:
                tensorboard_writer.add_scalar(f"{name}/{stage.lower()}", metrics_dict[name], epoch)
            metric_fct.reset()
        tepoch.set_postfix(loss=epoch_loss, metrics=metrics_dict)

    def compute_metrics(predictions, labels):
        if not metrics:
            tepoch.set_postfix(loss=loss.item())
            sleep(0.001)
            return
        metrics_dict = {}
        if any(predictions.sum(dim=1) != 1.0):
            predictions = softmax(predictions, dim=1)
        for name, metric_fct in metrics.items():
            metrics_dict[name] = f"{metric_fct(predictions.argmax(dim=1).to('cpu'), labels.argmax(dim=1).to('cpu')).item():.4f}"
        tepoch.set_postfix(loss=loss.item(), metrics=metrics_dict)
        sleep(0.001)

    model = model.to(device)

    train_epoch_len = len(train_loader)
    if validation_loader:
        validation_epoch_len = len(validation_loader)

    best_loss = 1e10
    best_loss_epoch = -1

    for epoch in range(epochs):
        model.train(True)
        start_time = time()

        print("-" * 10)
        print(f"Epoch {epoch + 1}/{epochs}")

        epoch_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                tepoch.set_description(f"Training data")
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                compute_metrics(outputs, labels)

            epoch_loss /= train_epoch_len
            if tensorboard_writer:
                tensorboard_writer.add_scalar("loss/train", epoch_loss, epoch)
            aggregate_metrics_per_epoch("Train")

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            epoch_loss = 0.0

            with tqdm(validation_loader, unit="batch") as tepoch:
                for val_batch_data in tepoch:
                    tepoch.set_description(f"Validation data")
                    val_images, val_labels = val_batch_data["image"].to(device), val_batch_data["label"].to(device)
                    with no_grad():
                        val_outputs = model(val_images)
                        loss = loss_function(val_outputs, val_labels)
                        epoch_loss += loss.item()
                        compute_metrics(val_outputs, val_labels)

                epoch_loss /= validation_epoch_len
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("loss/validation", epoch_loss, epoch)
                aggregate_metrics_per_epoch("Validation")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_loss_epoch = epoch + 1

                    if save_weights_dir:
                        save(model.state_dict(), join(save_weights_dir, f"{experiment_name}.pth"))

                    if verbose:
                        print(f"Lower loss: {best_loss:.4f} at epoch {best_loss_epoch}. Saved new best metric model.")

        epoch_elapsed_time = time() - start_time
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
