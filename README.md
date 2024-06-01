# Classify.py接口类文件说明
```
#classify.py源代码如下
import model
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ViolenceClass:
    def __init__(self):
        self.VioCL = model.ViolenceClassifier.load_from_checkpoint("../NIS4307/train_logs/resnet18_pretrain_test/version_2/checkpoints/resnet18_pretrain_test-epoch=20-val_loss=0.03.ckpt")  #加载模型
        self.VioCL.to(device = 'cuda:0')  #指定设备
        self.trainer = Trainer(accelerator='gpu', devices=[0])

    def classify(self, imgs : torch.Tensor):
        batchsize = imgs.size(0)  #获取n，即batch size
        preds = [-1]*batchsize  #初始化输出列表
        pred_dataloader = DataLoader(imgs, batch_size=batchsize)  #初始化pred_dataloader
        with torch.no_grad():  #固定参数
            prediction_scores = self.trainer.predict(self.VioCL,dataloaders=pred_dataloader,return_predictions=True)
        prediction_scores = prediction_scores[0]  #消除格式问题
        i = 0
        for cls_score in prediction_scores:  #返回的是一对值，分别是非暴力/暴力的得分，分类结果为得分高的一类
            if cls_score[0] > cls_score[1]:  
                preds[i] = 0
            else:
                preds[i] = 1
            i += 1
        return preds  #返回预测列表
```
## Introductions
创建ViolenceClass实例：
```
import classify
import torch
cls = classify.ViolenceClass()
```
指定的输入格式是n\*3\*224*224的torch.tensor，默认以pt格式传入
```
#加载输入向量，返回结果，并打印
loaded_tensor = torch.load('../NIS4307/vio_input_tensor.pt')
results = cls.classify(imgs = loaded_tensor.float())
print(results)
```
输出的结果应当是：
![4046624786fd09d6a2ea5a83a7fd6de](https://github.com/Arcs-ur/NIS4307_AI/assets/121781081/79239812-0978-42fc-93d2-b329cd0de99d)
**在运行使用者自己的输入之前，可以先使用```NIS4307/demo.py```进行测试，如成功则可自行调用classify.py**
# test.py文件使用说明
如果要直接使用test.py文件对各个数据集进行测试，需要把测试的数据集放进test文件夹中，并确保NIS4307/Violence_224文件下同时存在test、val、train三个文件夹（里面有没有内容不影响）。
