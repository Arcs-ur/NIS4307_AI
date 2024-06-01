#image reshape code 
import os
from pathlib import Path
from PIL import Image

# 设置输入和输出文件夹路径
input_folder = '/home/borui/chennan/NIS4307_AI/NIS4307/violence_224/暴力/暴力/'  #请根据具体情况做修改
output_folder = '/home/borui/chennan/NIS4307_AI/NIS4307/violence_224/baolireshape/' #请根据具体情况做修改

# 确保输出文件夹存在
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 打开图像文件
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # 调整图像大小为 224x224
        resized_image = image.resize((224, 224))

        # 保存压缩后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        resized_image.save(output_path)
        print(f'Saved {filename} to {output_folder}')