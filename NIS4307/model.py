import torch
from torch import nn
from torch import tensor
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAUROC

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet18(pretrained=True) #这里采用预训练过的模型 resnet18
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes) #这两行是修改最后的FCL的维数的
        # self.model = models.resnet18(pretrained=False, num_classes=2)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.auroc = MulticlassAUROC(num_classes = 2)

    def forward(self, x):
        return self.model(x) #前向传播

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch #获取batch的数据和标签
        logits = self(x) #前向传递，得到结果
        loss = self.loss_fn(logits, y) #得到结果后，用实际标签和预测结果来计算loss
        self.log('train_loss', loss) #记录loss
        return loss

    def validation_step(self, batch, batch_idx): 
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y) #还计算了预测的ACC
        self.log('val_loss', loss)
        self.log('val_acc', acc) #记录预测的ACC
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        acc = self.accuracy(logits, y) 
        auc = self.auroc(logits,y)
        self.log('test_acc', acc)
        self.log('loss',loss)
        self.log('AUROC',auc)
        return loss