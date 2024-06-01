import torch
import numpy as np
from PIL import Image
file_root = '/home/borui/chennan/NIS4307_AI/NIS4307/violence_224/test/' #请根据具体情况做修改
#假设你有 10 个图片文件 1_0000.jpg, 1_0001.jpg, ..., 1_0009.jpg
image_files = [
    file_root+'1_0010.png',
    file_root+'1_0001.png',
    file_root+'1_0002.png',
    file_root+'1_0003.png',
    file_root+'1_0004.png',
    file_root+'1_0005.png',
    file_root+'1_0006.png',
    file_root+'1_0007.png',
    file_root+'1_0008.png',
    file_root+'1_0009.png'
    # 添加更多图片...也可以直接对文件夹操作
]
# 创建一个空列表来存储图片张量
image_list = []

# 遍历图片文件,使用 PIL 读取并转换为 PyTorch 张量
for image_file in image_files:
    image = Image.open(image_file)
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    image_list.append(image_tensor)

# 使用 torch.stack() 函数将列表转换为 PyTorch 张量
input_tensor = torch.stack(image_list, dim=0)
input_tensor = input_tensor.to(torch.float32) / 255.0
# 检查张量的形状
torch.save(input_tensor,'/home/borui/chennan/NIS4307_AI/NIS4307/AIGC_vio.pt') #请根据具体情况做修改
