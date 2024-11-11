import os
from PIL import Image

# 定义存储数据的文件路径
output_file = 'image_sizes_all.txt'

# 定义数据文件夹路径
data_folder = r'D:\12045\adv-SR\imagenet_train_10000\imagenet_train_10000\train'

# 打开文件准备写入
with open(output_file, 'w') as f:
    # 遍历data文件夹中的所有文件
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # 只处理图片文件（根据扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                try:
                    # 打开图像文件
                    with Image.open(image_path) as img:
                        # 获取图像的宽度和高度
                        width, height = img.size

                        # 写入文件，每行记录图片的路径和大小
                        f.write(f"{image_path} -> {width}x{height}\n")
                except Exception as e:
                    print(f"无法处理 {image_path}: {e}")

print(f"图像尺寸信息已保存到 {output_file}")
