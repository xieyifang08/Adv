import matplotlib.pyplot as plt

# 读取 results.txt 文件并提取 Loss 数据
file_path = r'C:\Users\xieyi\Downloads\all_losses.txt'
import matplotlib.pyplot as plt

# 读取 results.txt 文件并提取 Loss 数据
# file_path = 'results.txt'
steps_list = []
losses = []

# 假设每个 epoch 有 50 个 batch
batch_per_epoch = 50

# 读取文件中的每一行数据
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 2:
            loss_str = parts[1].strip()
            try:
                # 提取 loss 值
                loss_value = float(loss_str.split('=')[1].strip())
                losses.append(loss_value)

                # 提取当前的 epoch 和 batch
                step_str = parts[0].strip()
                batch_info = step_str.split(', Batch')
                epoch = int(batch_info[0].split(' ')[1])  # 获取 epoch 编号
                batch = int(batch_info[1].strip())  # 获取 batch 编号

                # 计算全局 step
                step = (epoch - 1) * batch_per_epoch + batch
                steps_list.append(step)
            except Exception as e:
                print(f"无法解析这一行: {line} 错误: {e}")

# 绘制 step vs loss 的曲线
plt.figure(figsize=(10, 6))

# 绘制损失曲线
plt.plot(steps_list, losses, marker='o', linestyle='-', color='b', label='Loss')

# 添加标题和标签
plt.title('Loss Curve (Step vs Loss)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 显示图像
plt.show()
# 绘制 step vs loss 的曲线
# plt.figure(figsize=(10, 6))
#
# # 绘制损失曲线
# plt.plot(steps_list, losses, marker='o', linestyle='-', color='b', label='Loss')
#
# # 设置 y 轴范围
# plt.ylim(0.01, max(losses))  # 纵坐标从 0.01 开始
#
# # 添加标题和标签
# plt.title('Loss Curve (Step vs Loss)')
# plt.xlabel('Step')
# plt.ylabel('Loss')
#
# # 设置 y 轴的刻度
# plt.yticks([i * 0.01 for i in range(1, int(max(losses)/0.01) + 1)])
#
# # 添加网格线和图例
# plt.grid(True)
# plt.legend()
#
# # 显示图像
# plt.show()
#
#
