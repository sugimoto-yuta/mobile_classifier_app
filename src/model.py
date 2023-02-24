import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import mnasnet1_0
from torchvision import transforms

# 前処理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = mnasnet1_0(pretrained=True)
        self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h