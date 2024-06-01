from pytorch_lightning import Trainer
from model import ViolenceClassifier
from dataset import CustomDataModule
import pytorch_lightning
import torch
gpu_id = [0]
batch_size = 128
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
ckpt_root = "../NIS4307"
ckpt_path = ckpt_root + "/train_logs/resnet18_pretrain_test/version_2/checkpoints/resnet18_pretrain_test-epoch=20-val_loss=0.03.ckpt"
logger = pytorch_lightning.loggers.TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(accelerator='gpu', devices=gpu_id)
with torch.no_grad():
    trainer.test(model, data_module) 
