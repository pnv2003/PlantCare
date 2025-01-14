import os
import subprocess
import zipfile

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class PlantVillageDataset:
    
    @staticmethod
    def download():
        # if downloaded
        if os.path.exists('PlantVillage'):
            print("Data already downloaded!")
            return

        os.makedirs('PlantVillage', exist_ok=True)

        # download the dataset
        subprocess.run(["curl", "-L", "-o", "new-plant-diseases-dataset.zip", "https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset"])

        # unzip the dataset to the data directory
        with zipfile.ZipFile('new-plant-diseases-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('PlantVillage')

        # remove the zip file
        os.remove('new-plant-diseases-dataset.zip')

        print("Data downloaded successfully!")

    @staticmethod
    def prepare():

        root = 'PlantVillage/augmented'

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root=root, transform=transform)

        train_size = int(0.8 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        loaders = {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
        }

        return loaders
        
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
