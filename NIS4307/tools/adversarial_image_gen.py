#adversarial_image_generation with PGD algorithm
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

# 设置设备(CPU或GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.to(device)
model.eval()

# 设置文件夹路径
original_images_dir = '../NIS4307/violence_224/original_images' # 这里需要修改为你的图片目录
adversarial_images_dir = '../NIS4307/violence_224/adversarial_images' # 这里需要修改为你的图片目录

# 确保输出文件夹存在
os.makedirs(adversarial_images_dir, exist_ok=True)

def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40) :
    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        # 图像 + 梯度得到对抗样本
        adv_images = images + alpha*images.grad.sign()
        # 限制扰动范围
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# 遍历原始图像文件夹
for filename in os.listdir(original_images_dir):
    # 加载图像并预处理
    img_path = os.path.join(original_images_dir, filename)
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    X = transform(img).unsqueeze(0).to(device)
    label = torch.tensor(int(img_path.split("/")[-1][0])).unsqueeze(0).to(device)
    
    # 获取图像的真实标签

    # 生成对抗样本
    X_adv = pgd_attack(model, X, label, eps=0.1, alpha=0.01, iters=40)
    
    # 保存对抗样本图像
    adv_img = transforms.ToPILImage()(X_adv.squeeze(0).cpu())
    adv_img_path = os.path.join(adversarial_images_dir, filename)
    adv_img.save(adv_img_path)

print("对抗样本图像已成功保存至 'adversarial_images' 文件夹.")