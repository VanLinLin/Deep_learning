from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from utils.engine import test_step
import torchvision
from torch import nn

class CustomDataSet(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(Fr"D:\Van\Deep_learning\Data\test_images\{self.df.ID[index]}")
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

csv_file_path = r'D:\Van\Deep_learning\Data\test.csv'
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Lets create an object from our custom dataset class
test_data_object = CustomDataSet(csv_file_path, test_transform)

# Now lets use Data loader to load the data in batches
test_loader = torch.utils.data.DataLoader(
        test_data_object,
        batch_size=1
    )


resnet = torchvision.models.resnet152()

resnet_fc_in = resnet.fc.in_features

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

resnet.fc = nn.Linear(resnet_fc_in, 6)
resnet.load_state_dict(torch.load(r'D:\Van\Deep_learning\runs\ResNet\202310232444\weights\minimun_valid_loss_0.0327_epoch_28.pth'))

test_step(model=resnet,
           dataloader=test_loader,
           csv_file=r'D:\Van\Deep_learning\Data\test.csv',
           device=device)

