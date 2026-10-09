import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from src.training.dataset_loader import list_images_class
from datetime import datetime
from src.utils.config import (
    EMOTION_LABELS, EPOCHS, LEARNING_RATE,
    BATCH_SIZE, TRAINING_DATA_PATH
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class FERDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = image[:, :, ::-1].copy()
        image = image.astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN)/IMAGENET_STD

        return image, label

def set_dataloader(images, labels, batch_size, shuffle = True):
    dataset = FERDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = shuffle)
    return dataloader

def build_model(num_classes = 5):
    model = models.mobilenet_v2(weights = "IMAGENET1K_V1")
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model

def train_model(model, train_loader, epochs, learning_rate):
    device = torch.device("CUDA" if torch.cuda.is_available() else "CPU")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimazer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loses = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = images.to(device)

            optimazer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimazer.step()

            running_loss += loss.item()

        avg_loss = running_loses / len(train_loader)
        print(f"Epoch: {epoch + 1} / {epochs} - Loss: {avg_loss:.4f}")

    return model


def main():
    images, labels = list_images_class(TRAINING_DATA_PATH)
    print(f"Images uploeaded: {len(images)}")

    train_loader = set_dataloader(images, labels, batch_size = BATCH_SIZE, suffle = True)

    model = build_model(num_classes = len(EMOTION_LABELS))
    model = train_model(model, train_loader, epochs = EPOCHS, learning_rate = LEARNING_RATE)

    os.makedirs(models, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"models/fer_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Modelo guardado en models")
    with open(f"models/registro.txt", "a", encoding="utf-8") as f:
        f.write(
            f"Path: {model_path} | Images: {len(images)} | "
            f"Epochs: {EPOCHS} | lr: {LEARNING_RATE} | batch: {BATCH_SIZE}\n"
        )

if __name__ == "__main__" :
    main()