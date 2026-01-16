import os
import argparse
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from utils import get_loops, get_dataset_med, get_network, load_or_train_model
import cv2
from matplotlib import pyplot as plt

# 全局配置
MODELS = ['ConvNet', 'ResNet18']
CAM_METHODS = ['GradCAM', 'GradCAM++', 'XGradCAM']
SAVE_RATIO = 0.1  # 保存比例
HEATMAP_SIZE = (224, 224)  # 统一热力图尺寸


def main():
    # 初始化参数
    args = init_args()

    # 加载数据集
    channel, im_size, num_classes, class_names, mean, std, _, _, _, trainloader = get_dataset_med(
        args.dataset, args.data_path)

    # 预训练所有模型组合
    model_dict = pre_train_models(args, channel, num_classes, im_size, trainloader)

    # 处理数据集并生成可视化结果
    process_all_images(args, model_dict, trainloader, mean, std)


def init_args():
    parser = argparse.ArgumentParser(description='CAM Analysis')
    # parser.add_argument('--dataset', type=str, default='CIFAR10')
    # parser.add_argument('--data_path', type=str, default='data')
    # parser.add_argument('--save_root', type=str, default='./CAM_Results')
    # parser.add_argument('--batch_train', type=int, default=256)
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')\
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='oct', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_root', type=str, default='save_root_21new', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--proxy_model', type=str, default='ConvNet', help='proxy model for uncertainty estimation')
    parser.add_argument('--epoch_proxy', type=int, default=1, help='epochs for training the CAM proxy')
    # parser.add_argument('--epoch_proxy', type=int, default=1, help='epochs for training the CAM proxy')
    parser.add_argument('--save_model_path', type=str, default='./Model_Saved_new',
                        help='proxy model for uncertainty estimation')
    parser.add_argument('--validation_step', type=int, default=1, help='validation interval')
    parser.add_argument('--temperature', type=float, default=0.5, help='validation interval')
    parser.add_argument('--mode', type=str, default='CAMDM')
    parser.add_argument('--cam_path', type=str, default='./CAM_dir_new_Conv20', help='dataset path')
    parser.add_argument('--cam_type', type=str, default='GradCAM', help='dataset path')
    parser.add_argument('--checkpoint_path', type=str, default='./CAM_CKPT_BASE', help='dataset path')
    return parser.parse_args()


def pre_train_models(args, channel, num_classes, im_size, trainloader):
    """预训练所有模型组合并返回模型字典"""
    model_dict = {}
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model_name in MODELS:
        for cam_type in CAM_METHODS:
            key = f"{model_name}_{cam_type}"
            print(f">>> Training {key}...")

            # 初始化模型
            model = get_network(model_name, channel, num_classes, im_size).to(args.device)

            # 训练/加载模型
            args.proxy_model = model_name
            args.cam_type = cam_type
            args.checkpoint_path = f"./Model_Saved/{key}.pth"
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

            model = load_or_train_model(model, args, trainloader)
            model.eval()

            # 初始化CAM
            target_layer = model.features[-1] if model_name == 'ConvNet' else model.layer4[-1]
            # target_layer = model.module.features[-1] if model_name == 'ConvNet' else model.module.layer4[-1]
            cam = {
                'GradCAM': GradCAM,
                'GradCAM++': GradCAMPlusPlus,
                'XGradCAM': XGradCAM
            }[cam_type](model=model, target_layers=[target_layer])

            model_dict[key] = (model, cam)

    return model_dict


def process_all_images(args, model_dict, trainloader, mean, std):
    """处理所有图像并生成可视化结果"""
    # 初始化存储结构
    hist_matrix = np.zeros((len(MODELS), len(CAM_METHODS), 128))
    total_samples = len(trainloader.dataset)
    num_to_save = int(total_samples * SAVE_RATIO)

    # 创建保存目录
    os.makedirs(os.path.join(args.save_root, "composite"), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "originals"), exist_ok=True)

    # 处理每个批次
    saved_count = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(args.device)

        # 为所有模型生成CAM
        cam_results = {}
        for key, (model, cam) in model_dict.items():
            cam_output = cam(input_tensor=images)
            cam_results[key] = [postprocess_cam(cam) for cam in cam_output]

        # 处理每个样本
        for i in range(images.size(0)):
            if saved_count >= num_to_save:
                break

            # 保存原图
            original = denormalize_image(images[i], mean, std)
            original_path = os.path.join(args.save_root, "originals", f"img_{saved_count:05d}.png")
            cv2.imwrite(original_path, original)

            # 生成组合图
            composite = generate_composite(cam_results, i)
            composite_path = os.path.join(args.save_root, "composite", f"comp_{saved_count:05d}.png")
            cv2.imwrite(composite_path, composite)

            # 更新直方图统计
            update_histograms(hist_matrix, cam_results, i)

            saved_count += 1

        if saved_count >= num_to_save:
            break

    # 可视化直方图矩阵
    visualize_hist_matrix(hist_matrix)


def postprocess_cam(cam):
    """后处理CAM生成标准热力图"""
    cam_normalized = ((cam - cam.min()) / (cam.max() - cam.min()) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
    return cv2.resize(heatmap, HEATMAP_SIZE)


def denormalize_image(tensor, mean, std):
    """反归一化图像"""
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std) + mean  # 假设mean/std已经是numpy数组
    return (image * 255).astype(np.uint8)[..., ::-1]  # RGB转BGR


def generate_composite(cam_results, index):
    """生成6个子图的组合大图"""
    fig = plt.figure(figsize=(18, 12))

    for idx, (key, cams) in enumerate(cam_results.items()):
        ax = fig.add_subplot(2, 3, idx + 1)
        # ax.imshow(cv2.cvtColor(cams[index], cv2.COLOR_BGR2RGB))
        ax.imshow(cv2.cvtColor(cams[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(key, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    fig.canvas.draw()

    # 转换为OpenCV格式
    composite = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    composite = composite.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)


def update_histograms(hist_matrix, cam_results, index):
    """更新直方图统计"""
    for model_idx, model in enumerate(MODELS):
        for cam_idx, method in enumerate(CAM_METHODS):
            key = f"{model}_{method}"
            gray = cv2.cvtColor(cam_results[key][index], cv2.COLOR_BGR2GRAY)
            hist, _ = np.histogram(gray, bins=128, range=(0, 255))
            hist_matrix[model_idx, cam_idx] += hist


def visualize_hist_matrix(hist_matrix):
    """可视化直方图矩阵"""
    plt.figure(figsize=(18, 12))

    # 计算全局最大频率用于标准化
    max_freq = np.max(hist_matrix)

    for model_idx, model in enumerate(MODELS):
        for cam_idx, method in enumerate(CAM_METHODS):
            ax = plt.subplot(2, 3, model_idx * 3 + cam_idx + 1)
            hist_data = hist_matrix[model_idx, cam_idx]

            # 自动调整纵轴范围
            current_max = hist_data.max()
            y_max = max(current_max * 2.2, max_freq * 0.6)  # 确保当前最高值超过纵轴1/2

            plt.bar(range(128), hist_data, width=1.0)
            plt.ylim(0, y_max)
            plt.title(f"{model} - {method}", fontsize=10)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, 'histogram_matrix.png'))
    plt.close()


if __name__ == '__main__':
    main()