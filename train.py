import torch
import torchvision
import argparse
from torchvision import transforms
from torch import nn
from utils.engine import train, get_dataset_and_dataLoader, visualize, get_pretrained_resnet152
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--pretrained',
                        default=True,
                        help='Use pretrained weight.')
    parser.add_argument('--data_root',
                        default=r'D:/Van/Deep_learning/Data/',
                        help='The root path of data.')
    parser.add_argument('--num_classes',
                        default=6,
                        help='The number of classes of dataset. (Default: AOI dataset)')
    parser.add_argument('--epoch',
                        default=30,
                        help='Total train epochs.')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Visualize loss, accuracy, confusion metrix, etc.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.pretrained:

        resnet152 = get_pretrained_resnet152(num_classes=args.num_classes)

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=.5),
            transforms.RandomVerticalFlip(p=.5),
            transforms.ToTensor()
        ])
        valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        test_transform = valid_transform

        # Data setting
        data_root = Path(args.data_root)
        data = dict(
            train_data=dict(
                type='ImageFolder',
                path=data_root / Path('train'),
                transform=train_transform,
                batch_size=64
            ),
            valid_data=dict(
                type='ImageFolder',
                path=data_root / Path('valid'),
                transform=valid_transform,
                batch_size=1
            ),
            test_data=dict(
                type='CustomDataset',
                file=args.test_file,
                path=data_root / Path('test_images'),
                transform=test_transform,
                batch_size=1
            )
        )

        # Get dataLoader
        train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader = get_dataset_and_dataLoader(data_setting=data)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=resnet152.parameters(),
                                     lr=.001)

        # Train model
        result, save_path = train(model=resnet152,
                                  train_dataloader=train_dataloader,
                                  valid_dataloader=valid_dataloader,
                                  optimizer=optimizer,
                                  loss_function=loss_fn,
                                  epochs=int(args.epoch))

        # Check if need visualize
        if args.visualize:
            visualize(model=resnet152,
                      dataset=valid_dataset,
                      dataloader=valid_dataloader,
                      results=result,
                      save_path=save_path,
                      epochs=int(args.epoch))


if __name__ == "__main__":
    main()
