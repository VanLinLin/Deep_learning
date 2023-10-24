import torch
import pandas as pd
import datetime
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchvision import datasets, Module
from typing import Union, Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


# Define train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = 'cuda') -> Tuple[float, float]:
    """Train the model by running through each batch size and calculate loss, accuracy

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The train dataloader.
        loss_function (torch.nn.Module): The loss function to be use.
        optimizer (torch.optim.Optimizer): The optimizer to be use.
        device (torch.device, optional): The device which calculate the data. Defaults to 'cuda'.

    Returns:
        Tuple[float, float]: Return train loss, accuracy
    """
    # Set model to train mode
    model.train()

    # Put model in target device
    model.to(device=device)

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataLoader per batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_function(y_pred, y)
        train_loss += loss.item()

        # Optimize zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimize step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


# Define test step
def valid_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               device: torch.device = 'cuda') -> Tuple[float, float]:
    """Use the trained model to calculate the loss, accuracy in validation data by each epoch.

    Args:
        model (torch.nn.Module): The model to be used.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        loss_function (torch.nn.Module): The loss function to be use, same as train_step().
        device (torch.device, optional): The device which calculate the data. Defaults to 'cuda'.

    Returns:
        Tuple[float, float]: Return test loss, accuracy
    """
    # Set model to eval mode
    model.eval()

    # Put model in target device
    model.to(device=device)

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_function(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / \
                len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


# Define checkpoint saving method
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    """Save the specific checkpoint.

    Args:
        model (torch.nn.Module): The model to be used.
        target_dir (str): The saving file path.
        model_name (str): Checkpoint name.
    """

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = Path(target_dir) / Path(model_name)

    torch.save(obj=model.state_dict(),
               f=model_save_path)


# Combine train step and test step into train function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int,
          device: torch.device = 'cuda') -> Tuple[Dict[str, float], str]:
    """Train and valid the model, create the checkpoint and log folder and record the train and valid loss, accuracy.

    Args:
        model (torch.nn.Module): The model to be used.
        train_dataloader (torch.utils.data.DataLoader): Define each batch size and transformation of training data.
        valid_dataloader (torch.utils.data.DataLoader): Define each batch size and transformation of validation data. 
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        loss_function (torch.nn.Module): The loss function to be used.
        epochs (int): Total training epochs.
        device (torch.device, optional): The device which calculate the data.. Defaults to 'cuda'.

    Returns:
        Tuple[Dict[str, float], str]: Return result(dict) and saving path.
    """

    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []}

    # Setup saving path
    date = datetime.datetime.now()
    date_info = F"{date.year}{date.month}{date.day}{date.hour}{date.minute}{date.second}"
    model_name = model._get_name()
    print(model_name)
    save_path = Path(F"runs/{model_name}/{date_info}")
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = F"{save_path}/weights"
    log_path = F"{save_path}/logs/info.log"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(F"{save_path}/logs").mkdir(parents=True, exist_ok=True)
    Path(log_path).touch(exist_ok=True)

    min_valid_loss = 100
    with Path(log_path).open('a+') as f:
        f.write("Start training!\n")
        start_time = timer()
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                               dataloader=train_dataloader,
                                               loss_function=loss_function,
                                               optimizer=optimizer,
                                               device=device)

            valid_loss, valid_acc = valid_step(model=model,
                                               dataloader=valid_dataloader,
                                               loss_function=loss_function,
                                               device=device)

            message = F"[INFO] Epoch: {epoch + 1} | " \
                F"train loss: {train_loss: .4f} | " \
                F"train accuracy: {train_acc: .4f} | " \
                F"valid loss: {valid_loss: .4f} | " \
                F"valid accuracy: {valid_acc: .4f}"

            print(F"\n{message}\n")

            f.write(message)

            results["train_loss"].append(train_loss)
            results['train_acc'].append(train_acc)
            results['valid_loss'].append(valid_loss)
            results["valid_acc"].append(valid_acc)

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss

                print(
                    F"Saving best checkpoint with minimum valid loss {valid_loss:.4f} in {save_path}/weights")

                message += F"\nSaving best checkpoint with minimum valid loss {valid_loss:.4f}.\n"

                f.write(message)

                save_model(model=model,
                           target_dir=checkpoint_path,
                           model_name=F"minimun_valid_loss_{valid_loss:.4f}_epoch_{epoch + 1}.pth")

        end_time = timer()
        f.write(
            F"\n[INFO] Total training time: {end_time - start_time:.3f} seconds.")
        f.write("\nFininsh training!")
    torch.cuda.empty_cache()
    return results, save_path


# Define test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              csv_file: str = None,
              device: torch.device = 'cuda',
              save_path: str = None) -> None:
    """Inference the test data (without labels).

    Args:
        model (torch.nn.Module): The model to be used.
        dataloader (torch.utils.data.DataLoader): Define each batch size and transformation of test data. 
        csv_file (str): The file to be inferenced by model.
        device (torch.device, optional): The device which calculate the data.. Defaults to 'cuda'.
        save_path (str, optional): The saving path of inferenced file. Defaults to None.
    """
    # Put model in target device
    model.to(device)

    # Set model to eval mode
    model.eval()
    labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, X in tqdm(enumerate(dataloader)):
            # Send data to target device
            X = X.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate accuracy
            labels.append(test_pred_logits.argmax(dim=1).item())

    if csv_file:
        csv_df = pd.read_csv(csv_file)
        csv_df.Label = labels
        csv_df.to_csv(F"{save_path}/result.csv", index=False)


# Define custom dataset
def get_custom_dataset(csv_file: str, image_path: str, transform=Union[Module, None]) -> Dataset:
    """Get the custom dataset.

    Args:
        csv_file (str):The file to be inferenced by model.
        image_path (str): The certain path of test images in csv file.
        transform (_type_, optional): The transformation of test data. Defaults to Union[Module, None].

    Returns:
        Dataset: The customized test dataset. 
    """
    class CustomDataSet(Dataset):
        """Inherent the torch.utils.data.Dataset to create custom dataset.

        Args:
            Dataset (_type_): torch.utils.data.Dataset
        """

        def __init__(self, csv_file: str, image_path: str, transform=Union[Module, None]):
            self.df = pd.read_csv(csv_file)
            self.transform = transform
            self.image_path = image_path

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, index):
            image = Image.open(Fr"{self.image_path}/{self.df.ID[index]}")
            image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    return CustomDataSet(csv_file, image_path, transform)


# Define dataloader getter
def get_dataset_and_dataLoader(data_setting: dict) -> Tuple[Dataset, DataLoader, Dataset, DataLoader, Dataset, DataLoader]:
    """Get the train, valid, test dataloader.

    Args:
        data_setting (dict): The setting of data. Include path, transform, batchsize.
        e.g.
            data_root = Path('D:/Van/Deep_learning/Data/')
            data = dict(
                train_data=dict(
                    type = 'ImageFolder',
                    path = data_root / Path('train'),
                    transform = train_transform,
                    batch_size = 64
                ),
                valid_data = dict(
                    type = 'ImageFolder',
                    path = data_root / Path('valid'),
                    transform = valid_transform,
                    batch_size = 1
                ),
                test_data = dict(
                    type = 'CustomDataset',
                    file = r'D:/Van/Deep_learning/Data/test.csv',
                    path = data_root / Path('test_images'),
                    transform = test_transform,
                    batch_size = 1
                )
            )        

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Return train, valid and test dataloader.
    """
    train_dataloader, valid_dataloader, test_dataloader = [], [], []
    train_dataset, valid_dataset, test_dataset = [], [], []
    try:
        if data_setting['train_data']:
            if data_setting['train_data']['type'].casefold() == 'ImageFolder'.casefold():
                # Define dataset
                train_dataset = datasets.ImageFolder(root=data_setting['train_data']['path'],
                                                    transform=data_setting['train_data']['transform'])

                # Define dataloader
                train_dataloader = DataLoader(dataset=train_dataset,
                                            batch_size=data_setting['train_data']['batch_size'],
                                            shuffle=True,
                                            generator=torch.Generator(device='cpu'))
    except:
        assert ("Data setting error! Need train data settings.")

    try:
        if data_setting['valid_data']:
            if data_setting['valid_data']['type'].casefold() == 'ImageFolder'.casefold():
                # Define dataset
                valid_dataset = datasets.ImageFolder(root=data_setting['valid_data']['path'],
                                                    transform=data_setting['valid_data']['transform'])

                # Define dataloader
                valid_dataloader = DataLoader(dataset=valid_dataset,
                                            batch_size=data_setting['valid_data']['batch_size'])
    except:
        assert ("Data setting error! Need validation data settings.")

    try:
        if data_setting['test_data']:
            if data_setting['test_data']['type'].casefold() == 'ImageFolder'.casefold():
                # Define dataset
                test_dataset = datasets.ImageFolder(root=data_setting['test_data']['path'],
                                                    transform=data_setting['test_data']['transform'])

                # Define dataloader
                test_dataloader = DataLoader(dataset=test_dataset,
                                            batch_size=data_setting['test_data']['batch_size'])

            elif data_setting['test_data']['type'].casefold() == 'CustomDataset'.casefold():
                test_dataset = get_custom_dataset(csv_file=data_setting['test_data']['file'],
                                                image_path=data_setting['test_data']['path'],
                                                transform=data_setting['test_data']['transform'])

                # Define test dataLoader
                test_dataloader = DataLoader(dataset=test_dataset,
                                            batch_size=data_setting['test_data']['batch_size'])
    except:
        print("No test data setting.")

    return train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader


# plot confusion matrix
def draw_confmat(model: torch.nn.Module,
                 dataset: Dataset,
                 dataloader: torch.utils.data.DataLoader,
                 save_path: str,
                 device: torch.device = 'cuda'
                 ):
    model.to(device)

    # Set model to eval mode
    model.eval()
    pred_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            # Send data to target device
            X = X.to(device)

            # Forward pass
            test_pred_logits = model(X)

            pred_labels.append(test_pred_logits.argmax(dim=1).cpu().item())

    confmat = ConfusionMatrix(task='multiclass',
                              num_classes=len(dataset.classes))
    confmat_tensor = confmat(preds=torch.Tensor(pred_labels),
                             target=torch.Tensor(dataset.targets))

    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=dataset.classes,
                                    figsize=(len(dataset.classes),
                                             len(dataset.classes)),
                                    show_normed=True,
                                    colorbar=True)
    ax.set_title("Confusion matrix of validation data")
    fig.savefig(F"{save_path}/confusion_matrix.jpg")


# Visualize
def visualize(model: torch.nn.Module,
              dataset: Dataset,
              dataloader: torch.utils.data.DataLoader,
              results: Dict[str, float],
              save_path: str,
              epochs: int):
    vis_path = (Path(save_path) / Path('vis'))
    vis_path.mkdir(parents=True, exist_ok=True)

    # Setup figure size
    plt.figure(figsize=(15, 15))

    # Plot loss
    plt.subplot(121)
    plt.plot(range(epochs), results['train_loss'], label='train')
    plt.plot(range(epochs), results['valid_loss'], label='valid')
    plt.title("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(122)
    plt.plot(range(epochs), results['train_acc'], label='train')
    plt.plot(range(epochs), results['valid_acc'], label='valid')
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(F"{vis_path}/visualize.jpg")

    # Plot confusion matrix
    draw_confmat(model=model,
                 dataset=dataset,
                 dataloader=dataloader,
                 save_path=vis_path)


# Pretrained resnet152
def get_resnet152(num_classes: int, pretrained: bool = False):

    if pretrained:
        # Get pretrained checkpoint
        weights = torchvision.models.ResNet152_Weights.DEFAULT

        # Get resnet152 model with pretrained weights
        resnet152 = torchvision.models.resnet152(weights=weights)
    else:
        
        resnet152 = torchvision.models.resnet152()


    # Get the output channels before fc layer
    resnet152_fc_in = resnet152.fc.in_features

    # Set output channels into AOI classes
    resnet152.fc = nn.Linear(resnet152_fc_in, int(num_classes))

    return resnet152
