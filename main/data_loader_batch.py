import os

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ActionRecognitionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if not self.is_test:
            self.label_encoder = LabelEncoder()
            self.data_frame["encoded_label"] = self.label_encoder.fit_transform(
                self.data_frame["label"]
            )
        else:
            self.label_encoder = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if not self.is_test:
            label = self.data_frame.iloc[idx]["encoded_label"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image


def get_data_loaders(train_csv, test_csv, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # create datasets
    train_dataset = ActionRecognitionDataset(
        train_csv, os.path.join("..", "Human Action Recognition", "train"), train_transform
    )
    test_dataset = ActionRecognitionDataset(
        test_csv, os.path.join("..", "Human Action Recognition", "test"), test_transform, is_test=True
    )

    print(
        f"Total samples in original training dataset before splitting: {len(train_dataset)}"
    )

    # split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.dataset.label_encoder


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_encoder = get_data_loaders(
        os.path.join("..", "Human Action Recognition", "Training_set.csv"),
        os.path.join("..", "Human Action Recognition", "Testing_set.csv"),
    )
    
    print("\nDataset Summary:")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"\nClass labels: {label_encoder.classes_}")

    print("\nDataLoader Summary:")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels in a batch: {labels}")
    print(f"Label data type: {labels.dtype}")
    # Check a test batch
    # test_images = next(iter(test_loader))
    # print(f"\nTest batch shape: {test_images.shape}")

    print("\nData loading and preprocessing completed successfully.")
   