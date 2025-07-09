import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.efficientnet_model import EfficientLite
import os

def main():
    torch.manual_seed(42)

    DATA_DIR = "data/garbage_classification"
    assert os.path.exists(DATA_DIR), f"Dataset not found at {DATA_DIR}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, scale=(1, 2), shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # ✅ Safe config for Windows: NO multiprocessing, small batch size
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = EfficientLite(lr=3e-5, num_class=num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="best_model"
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

# ✅ Required for multiprocessing on Windows
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
