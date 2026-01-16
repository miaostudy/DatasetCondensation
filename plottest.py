import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# 简单的生成函数，产生从高斯分布生成的样本
def generate_data(num_points=500):
    # 从标准正态分布中采样点
    x = np.linspace(-5, 5, num_points)
    y = np.sin(x) + 0.3 * np.random.randn(num_points)  # 加一些噪声
    return x, y

# 定义扩散过程，逐步添加噪声
def add_noise(x, num_steps=10):
    noisy_data = []
    for step in range(num_steps):
        noise_level = step / num_steps  # 控制噪声的程度
        noisy_sample = x + np.random.randn(len(x)) * noise_level
        noisy_data.append(noisy_sample)
    return noisy_data

# 定义去噪过程（简化为反向去噪）
def denoise(noisy_data, num_steps=10):
    denoised_data = []
    for step in range(num_steps):
        noise_level = (num_steps - step) / num_steps  # 逐渐去噪
        denoised_sample = noisy_data[step] - np.random.randn(len(noisy_data[step])) * noise_level
        denoised_data.append(denoised_sample)
    return denoised_data

# 生成数据
x, y = generate_data()

# 添加噪声（模拟正向扩散过程）
noisy_data = add_noise(y)

# 进行反向去噪（模拟反向扩散过程）
denoised_data = denoise(noisy_data)

# 绘图展示
plt.figure(figsize=(12, 6))

# 原始数据
plt.subplot(1, 3, 1)
plt.plot(x, y, label='Original Data', color='black')
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')

# 添加噪声后的数据（正向扩散）
plt.subplot(1, 3, 2)
for i in range(0, len(noisy_data), len(noisy_data)//5):  # 每隔几个步骤绘制一次
    plt.plot(x, noisy_data[i], label=f'Noisy Step {i}')
plt.title('Forward Diffusion (Noise Added)')
plt.xlabel('X')
plt.ylabel('Y')

# 去噪后的数据（反向扩散）
plt.subplot(1, 3, 3)
for i in range(0, len(denoised_data), len(denoised_data)//5):  # 每隔几个步骤绘制一次
    plt.plot(x, denoised_data[i], label=f'Denoised Step {i}')
plt.title('Reverse Diffusion (Denoising)')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
