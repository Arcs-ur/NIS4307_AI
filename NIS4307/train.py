from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning
from model import ViolenceClassifier
from dataset import CustomDataModule
#确实是在这个文件里面进行训练的。

gpu_id = [0]
lr = 3e-4
batch_size = 64
log_name = "resnet18_pretrain_test"
print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))



data_module = CustomDataModule(batch_size=batch_size)
#接下来需要把数据集给初始化。
data_module.setup(stage=None)


# 设置模型检查点，用于保存最佳模型
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)
#相当于一个评估标准是在验证集上的loss的保存标准。

logger = pytorch_lightning.loggers.TensorBoardLogger("train_logs", name=log_name)



# 实例化训练器
trainer = Trainer(
    max_epochs=40,
    accelerator='gpu',
    devices=gpu_id,
    logger=logger,
    callbacks=[checkpoint_callback]
)



# 实例化模型
model = ViolenceClassifier(learning_rate=lr)

# 开始训练
trainer.fit(model = model,datamodule = data_module)
