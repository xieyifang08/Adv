import os
from PIL import Image

# 定义图像尺寸信息文件路径
size_file = 'image_sizes_all.txt'

# 定义存储处理后图片的目标文件夹
output_folder = r'D:\adv\PLP\adv\attack_PGD_alexnet_resize_all'

# 定义原图文件夹
input_folder = r'D:\12045\adv-SR\PLP\fast_adv_imagenet\attacks\advs\pgd_alexnet'  # 替换为实际的文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取图像尺寸信息并进行resize操作
with open(size_file, 'r') as f:
    for line in f:
        # 解析每一行的信息
        parts = line.strip().split(' -> ')
        if len(parts) == 2:
            image_name = os.path.basename(parts[0])  # 获取图像文件名
            size_str = parts[1]

            # 获取原始尺寸 (宽x高)
            try:
                width, height = map(int, size_str.split('x'))

                # 构建图像的完整路径
                image_path = os.path.join(input_folder, image_name)

                # 检查图片是否存在
                if os.path.exists(image_path):
                    # 打开原图
                    with Image.open(image_path) as img:
                        # 按照原尺寸进行resize
                        resized_img = img.resize((width, height))

                        # 获取图片文件名并拼接输出路径
                        output_path = os.path.join(output_folder, image_name)

                        # 保存处理后的图片
                        resized_img.save(output_path)
                        print(f"已保存: {output_path}")
                else:
                    print(f"图片不存在: {image_path}")
            except Exception as e:
                print(f"处理图片 {image_name} 时出错: {e}")

print("所有图片已按照原尺寸调整并保存。")

