import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from utils import load_or_train_model, train_CAM, get_loops, get_dataset, get_dataset_med, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--proxy_model', type=str, default='ResNet18', help='proxy model for uncertainty estimation')
    parser.add_argument('--epoch_proxy', type=int, default=1, help='epochs for training the CAM proxy')
    # parser.add_argument('--epoch_proxy', type=int, default=1, help='epochs for training the CAM proxy')
    parser.add_argument('--save_model_path', type=str, default='./Model_Saved_new',
                        help='proxy model for uncertainty estimation')
    parser.add_argument('--validation_step', type=int, default=1, help='validation interval')
    parser.add_argument('--temperature', type=float, default=0.5, help='validation interval')
    parser.add_argument('--mode', type=str, default='CAMDM')
    parser.add_argument('--cam_path', type=str, default='./CAM_dir_new', help='dataset path')
    parser.add_argument('--cam_type', type=str, default='GradCAM', help='dataset path')
    parser.add_argument('--checkpoint_path', type=str, default='./CAM_CKPT', help='dataset path')


    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False



    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    args.checkpoint_path = args.checkpoint_path + '/' + args.proxy_model + '_' + args.cam_type + '.pth'
    print('CKPT_path: ',args.checkpoint_path)
    #load DATA
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_dataset_med(args.dataset, args.data_path)

    # 统计像素直方图
    pixel_histogram = np.zeros(256)



    #load CAM
    ''' Train CAM proxy'''
    net_cam = get_network(args.proxy_model, channel, num_classes, im_size).to(args.device)  # get a random model

    net_cam.eval()
    net_cam = load_or_train_model(net_cam, args, trainloader)
    # 选择目标层（通常是最后一个卷积层）
    if args.proxy_model == 'ConvNet':
        target_layer = net_cam.features[-1]  # 最后一个卷积层之前的激活层
    elif args.proxy_model == 'ResNet18':
        target_layer = net_cam.layer4[-1]
    # 初始化 GradCAM，移除 use_cuda 参数
    if args.cam_type == 'GradCAM':
        cam_model = GradCAM(model=net_cam, target_layers=[target_layer])
    elif args.cam_type == 'GradCAM++':
        cam_model = GradCAMPlusPlus(model=net_cam, target_layers=[target_layer])
    elif args.cam_type == 'XGradCAM':
        cam_model = XGradCAM(model=net_cam, target_layers=[target_layer])

    #Statistic
    for batch in trainloader:
        images, _ = batch
        images = images.to(args.device)

        cams = cam_model(input_tensor=images)

        for cam in cams:
            cam_normalized = ((cam - cam.min()) / (cam.max() - cam.min()) * 255).astype(np.uint8)
            hist, _ = np.histogram(cam_normalized, bins=256, range=(0, 255))
            pixel_histogram += hist
    # 可视化直方图
    plt.bar(range(256), pixel_histogram, color='blue', alpha=0.7)
    plt.title('Pixel Distribution Histogram of CAMs')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    main()
