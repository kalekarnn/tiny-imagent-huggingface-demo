from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size):
    dataset = load_dataset("Maysee/tiny-imagenet")
    print(f"Dataset keys: {dataset.keys()}")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class TinyImageNetDataset:
        def __init__(self, dataset_split, transform):
            self.data = dataset_split
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]["image"]
            label = self.data[idx]["label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_loader = DataLoader(
        TinyImageNetDataset(dataset["train"], transform_train), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TinyImageNetDataset(dataset["valid"], transform_val), batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader
