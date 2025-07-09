import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch

class EfficientLite(pl.LightningModule):
    def __init__(self, lr, num_class):
        super().__init__()
        self.save_hyperparameters()

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_class)

        self.train_accuracy = MulticlassAccuracy(num_class)
        self.val_accuracy = MulticlassAccuracy(num_class)
        self.test_accuracy = MulticlassAccuracy(num_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(torch.argmax(logits, dim=1), y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(torch.argmax(logits, dim=1), y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.test_accuracy(torch.argmax(logits, dim=1), y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
