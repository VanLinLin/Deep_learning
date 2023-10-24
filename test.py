import torchvision
import pandas as pd
import torch
import argparse
from pathlib import Path
from torchvision import transforms
from utils.engine import test_step, get_resnet152, get_dataset_and_dataLoader



def parse_args():
    parser = argparse.ArgumentParser(description='Interence test data.')
    parser.add_argument('--checkpoint_path',
                        required=True,
                        help='THe chekcpoint path.')
    parser.add_argument('--data_root',
                        required=True,
                        help='The root path of data.')
    parser.add_argument('--csv_file_path',
                        required=True,
                        help='The test csv file.')
    parser.add_argument('--num_classes',
                        default=6,
                        help='The number of classes of dataset.')
    parser.add_argument('--result_save_path',
                        default=Path().cwd(),
                        help='The number of classes of dataset.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    resnet152 = get_resnet152(num_classes=args.num_classes)

    resnet152.load_state_dict(torch.load(args.checkpoint_path))

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
            file=args.csv_file_path,
            path=data_root / Path('test_images'),
            transform=test_transform,
            batch_size=1
        )
    )

    train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader = get_dataset_and_dataLoader(data_setting=data)

    test_step(model=resnet152,
           dataloader=test_dataloader,
           csv_file=args.csv_file_path,
           save_path=args.result_save_path)

if __name__ == "__main__":
    main()