import os

# 设置图片所在的目录
directory = '/home/borui/chennan/NIS4307_AI/NIS4307/violence_224/暴力/暴力' # 这里需要修改为你的图片目录

# 遍历目录中的所有文件
for i, filename in enumerate(os.listdir(directory), start=1):
    # 检查文件是否为图片
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 构建新的文件名
        new_filename = f'1_{i:04d}.jpg'
        # 构建完整的文件路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed {filename} to {new_filename}')