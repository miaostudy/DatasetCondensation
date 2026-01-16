import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from utils import train_CAM, get_loops, get_dataset, get_dataset_med, get_network, get_eval_pool, evaluate_synset, \
    get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, wavelet_transform, \
    inverse_wavelet_transform, get_optimizer
from pytorch_wavelets import DWTForward, DWTInverse
import logging

# 强制使用单个GPU，避免DataParallel问题
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 定义ResNet18的基础模块（确保forward方法正确）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 重新定义ResNet18类，使用AdaptiveAvgPool2d
class ResNet18_CAM(nn.Module):
    def __init__(self, num_classes=10, channel=3):
        super(ResNet18_CAM, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 使用AdaptiveAvgPool2d，确保输出1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 输出: [batch_size, 512, 4, 4] for CIFAR10

        x = self.avgpool(x)  # 输出: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)  # 输出: [batch_size, 512]
        x = self.fc(x)  # 输出: [batch_size, num_classes]

        return x


def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger


# 辅助函数：安全获取原始模型（处理DataParallel包装）
def get_original_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# 辅助函数：确保模型所有参数都在指定设备上
def ensure_model_on_device(model, device):
    model = model.to(device)
    # 确保模型权重是float32
    model = model.float()
    for param in model.parameters():
        param.data = param.data.to(device).float()
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device).float()
    for buf in model.buffers():
        buf.data = buf.data.to(device).float()
    return model


# 辅助函数：确保张量是float32类型
def ensure_float32(tensor, device):
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.to(device)


# CIFAR10类别名
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_and_save_confusion_matrix(cm, class_names, save_path, args, exp):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, args.method + '_' + str(exp) + '_confusion_matrix.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--log_dir', type=str, default='./log_0905')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--lr_proxy', type=float, default=1e-4, help='learning rate for CAM proxy')
    parser.add_argument('--lr_w', type=float, default=0.1)
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result_0905', help='path to save results')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='path to save model checkpoints')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--proxy_model', type=str, default='ResNet18', help='proxy model for uncertainty estimation')
    parser.add_argument('--epoch_proxy', type=int, default=5, help='epochs for training the CAM proxy')
    parser.add_argument('--save_model_path', type=str, default='./Model_Saved_new',
                        help='proxy model for uncertainty estimation')
    parser.add_argument('--validation_step', type=int, default=1, help='validation interval')
    parser.add_argument('--temperature', type=float, default=0.5, help='validation interval')
    parser.add_argument('--mode', type=str, default='CAMDM')
    parser.add_argument('--cam_path', type=str, default='./CAM_dir', help='CAM visualization path')
    parser.add_argument('--cam_type', type=str, default='GradCAM', help='CAM type')

    args = parser.parse_args()
    args.method = 'WCAMDM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    # 确保checkpoint目录存在
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # 为train_CAM函数设置checkpoint路径
    args.checkpoint_path = os.path.join(args.checkpoint_path, f'cam_proxy_{args.proxy_model}_{args.dataset}_exp0.pth')

    # 打印设备信息
    logger = build_logger('./log_1029', cfgname='temp_logger')
    logger.info(f'Using device: {args.device}')
    if args.device == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

    # 重新创建正式logger
    root = os.getcwd()
    log_name = args.mode + '_IPC_{}_Model_{}_Proxy_{}_Epoch_{}_Eval_{}_CAM_{}_lrw_{}_limg_{}_Data_{}'.format(args.ipc,
                                                                                                             args.model,
                                                                                                             args.proxy_model,
                                                                                                             args.epoch_proxy,
                                                                                                             args.eval_mode,
                                                                                                             args.cam_type,
                                                                                                             args.lr_w,
                                                                                                             args.lr_img,
                                                                                                             args.dataset)
    log_path = os.path.join(args.log_dir, log_name)
    logger = build_logger('./log_1029', cfgname=log_name)
    args.logger = logger

    # 确保所有目录存在
    for path in [args.log_dir, args.data_path, args.save_path, args.cam_path, args.checkpoint_path,
                 args.save_model_path]:
        if path and not os.path.exists(os.path.dirname(path) if '.' in os.path.basename(path) else path):
            os.makedirs(os.path.dirname(path) if '.' in os.path.basename(path) else path)

    eval_it_pool = np.arange(0, args.Iteration + 1,
                             2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]
    logger.info('eval_it_pool: {}'.format(eval_it_pool))
    logger.info('Will output evaluation metrics at iterations: {}'.format(eval_it_pool))

    # 加载数据集
    logger.info('Loading dataset {}...'.format(args.dataset))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_dataset_med(
        args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        # 更新exp相关的checkpoint路径
        args.exp = exp
        args.checkpoint_path = os.path.join('./checkpoints',
                                            f'cam_proxy_{args.proxy_model}_{args.dataset}_exp{exp}.pth')

        logger.info('\n================== Exp %d ==================\n ' % exp)
        logger.info('Hyper-parameters: \n{}'.format(args.__dict__))
        logger.info('Evaluation model pool: {}'.format(model_eval_pool))
        logger.info('CAM proxy checkpoint will be saved to: {}'.format(args.checkpoint_path))

        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        # 确保图像数据是float32
        images_all = torch.cat(images_all, dim=0).to(args.device)
        images_all = ensure_float32(images_all, args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            logger.info('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            imgs = images_all[idx_shuffle]
            # 确保返回的图像是float32
            return ensure_float32(imgs, args.device)

        for ch in range(channel):
            logger.info('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]),
                                                                             torch.std(images_all[:, ch])))

        # 初始化合成数据 - 明确指定dtype为float32
        label_syn = torch.repeat_interleave(torch.arange(num_classes, device=args.device), args.ipc)
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float32,  # 明确使用float32
                                requires_grad=True,
                                device=args.device)

        if args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                imgs = get_images(c, args.ipc).detach().data
                imgs = ensure_float32(imgs, args.device)
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = imgs
        else:
            logger.info('initialize synthetic data from random noise')

        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        logger.info('%s training begins' % get_time())

        ''' Train CAM proxy - 彻底修复部分 '''
        logger.info('=' * 50)
        logger.info('Training CAM proxy model: {}'.format(args.proxy_model))
        logger.info('Proxy model training epochs: {}'.format(args.epoch_proxy))
        logger.info('Proxy model learning rate: {}'.format(args.lr_proxy))

        # 直接使用我们自定义的ResNet18_CAM，确保forward方法正确
        if args.proxy_model == 'ResNet18':
            net_cam = ResNet18_CAM(num_classes=num_classes, channel=channel)
            logger.info('Using custom ResNet18_CAM with AdaptiveAvgPool2d')
        else:
            # 对于其他模型，使用原有的get_network
            net_cam = get_network(args.proxy_model, channel, num_classes, im_size)

        # 确保模型在正确设备上且为float32
        net_cam = ensure_model_on_device(net_cam, args.device)
        net_cam = net_cam.float()  # 强制模型为float32
        original_net_cam = get_original_model(net_cam)

        # 验证池化层是否正确
        if hasattr(original_net_cam, 'avgpool'):
            logger.info(f'Pooling layer type: {type(original_net_cam.avgpool)}')
            if isinstance(original_net_cam.avgpool, nn.AdaptiveAvgPool2d):
                logger.info(f'AdaptiveAvgPool2d output size: {original_net_cam.avgpool.output_size}')
            else:
                logger.warning(f'Unexpected pooling layer: {type(original_net_cam.avgpool)}')
                # 强制替换为AdaptiveAvgPool2d
                original_net_cam.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(args.device).float()

        net_cam.train()
        logger.info('Starting CAM proxy training...')
        train_CAM(model=net_cam, trainloader=trainloader, args=args)
        logger.info('CAM proxy training finished!')

        # 选择目标层（使用原始模型）
        try:
            if args.proxy_model == 'ConvNet':
                target_layer = original_net_cam.features[-1] if hasattr(original_net_cam, 'features') else \
                original_net_cam.layer4[-1]
            elif args.proxy_model == 'ResNet18':
                # ResNet18_CAM的目标层是layer4的最后一个BasicBlock
                target_layer = original_net_cam.layer4[-1]
            else:
                target_layer = original_net_cam.layer4[-1] if hasattr(original_net_cam, 'layer4') else \
                original_net_cam.features[-1]

            # 确保目标层在正确设备上且为float32
            target_layer = target_layer.to(args.device).float()
            logger.info(f'Selected CAM target layer: {target_layer.__class__.__name__}')
        except Exception as e:
            logger.error(f'Error selecting target layer: {str(e)}')
            raise

        # 初始化 GradCAM - 修复targets格式问题
        try:
            if args.cam_type == 'GradCAM':
                cam = GradCAM(model=net_cam, target_layers=[target_layer])
            elif args.cam_type == 'GradCAM++':
                cam = GradCAMPlusPlus(model=net_cam, target_layers=[target_layer])
            elif args.cam_type == 'XGradCAM':
                cam = XGradCAM(model=net_cam, target_layers=[target_layer])
            logger.info(f'Initialized {args.cam_type} successfully (use_cuda={args.device == "cuda"})')
        except Exception as e:
            logger.error(f'Error initializing CAM: {str(e)}')
            raise

        # 定义小波变换模块 - 确保为float32
        DWT = DWTForward(J=1, wave='haar').to(args.device).float()
        IDWT = DWTInverse(wave='haar').to(args.device).float()
        logger.info('Wavelet transform modules initialized (float32)')

        # 开始主训练循环
        logger.info('=' * 50)
        logger.info('Starting main training loop (total iterations: {})'.format(args.Iteration))
        logger.info('Will evaluate at iterations: {}'.format(eval_it_pool))

        cam_error_count = 0
        for it in range(args.Iteration + 1):
            # 每100次迭代打印一次进度
            if it % 100 == 0 and it != 0:
                logger.info(
                    'Progress: {}/{} iterations ({}%)'.format(it, args.Iteration, int(it / args.Iteration * 100)))
                if cam_error_count > 0:
                    logger.warning(f'CAM generation errors so far: {cam_error_count}')
                    cam_error_count = 0

            if it % 1000 == 0:
                logger.info('=' * 30)
                logger.info('Epoch: {}'.format(it))

            ''' Evaluate synthetic data '''
            save_path_cms = './cms_' + args.method
            if it in eval_it_pool:
                logger.info('=' * 50)
                logger.info('Starting evaluation at iteration {}'.format(it))
                logger.info('This may take some time (evaluating {} random models)...'.format(args.num_eval))

                for model_eval in model_eval_pool:
                    logger.info(
                        '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                            args.model, model_eval, it))
                    logger.info('DSA augmentation strategy: \n{}'.format(args.dsa_strategy))
                    logger.info('DSA augmentation parameters: \n{}'.format(args.dsa_param.__dict__))

                    accs = []
                    sens = []
                    spes = []
                    f1s = []
                    cms = []
                    for it_eval in range(args.num_eval):
                        if it_eval % 5 == 0:  # 每5个模型打印一次进度
                            logger.info('Evaluating model {}/{}'.format(it_eval + 1, args.num_eval))

                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        net_eval = net_eval.float()  # 确保评估模型也是float32
                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        image_syn_eval = ensure_float32(image_syn_eval, args.device)
                        label_syn_eval = copy.deepcopy(label_syn.detach())

                        _, acc_train, acc_test, sensitivity, specificity, f1, cm = evaluate_synset(it_eval, net_eval,
                                                                                                   image_syn_eval,
                                                                                                   label_syn_eval,
                                                                                                   testloader, args)
                        accs.append(acc_test)
                        sens.append(sensitivity)
                        spes.append(specificity)
                        f1s.append(f1)
                        cms.append(cm)

                    # 打印评估结果
                    logger.info('=' * 30)
                    logger.info('Evaluation Results at iteration {}'.format(it))
                    logger.info('Evaluate %d random %s' % (len(accs), model_eval))
                    logger.info('ACCmean = %.4f ACCstd = %.4f' % (np.mean(accs), np.std(accs)))
                    logger.info('SENmean = %.4f SENstd = %.4f' % (np.mean(sens), np.std(sens)))
                    logger.info('SPEmean = %.4f SPEstd = %.4f' % (np.mean(spes), np.std(spes)))
                    logger.info('F1mean = %.4f F1std = %.4f' % (np.mean(f1s), np.std(f1s)))
                    logger.info('=' * 30)

                    if it == eval_it_pool[-1]:
                        combined_cm = np.sum(cms, axis=0)
                        plot_and_save_confusion_matrix(combined_cm, class_names, save_path_cms, args, exp)

                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path,
                                         'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' % (args.method, args.dataset, args.model,
                                                                                  args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                image_syn_vis = ensure_float32(image_syn_vis, 'cpu')
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
                save_image(image_syn_vis, save_name, nrow=args.ipc)
                logger.info('Synthetic images saved to: {}'.format(save_name))

            ''' Train target network'''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net = net.float()  # 确保目标网络是float32
            net.train()
            original_net = get_original_model(net)

            '''Freeze Net and optimize images'''
            for param in list(net.parameters()):
                param.requires_grad = False

            # 获取embed层
            embed = original_net.embed if hasattr(original_net, 'embed') else original_net.features

            loss_avg = 0
            if 'BN' not in args.model:
                loss = torch.tensor(0.0, dtype=torch.float32, device=args.device)
                cam_list = []

                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_real = ensure_float32(img_real, args.device)

                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1]))
                    img_syn = ensure_float32(img_syn, args.device)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        # DSA后再次确保float32
                        img_real = ensure_float32(img_real, args.device)
                        img_syn = ensure_float32(img_syn, args.device)

                    ''' Gen CAM for img_syn - 修复两个关键错误 '''
                    try:
                        # 限制batch size，避免内存溢出
                        batch_size_cam = min(32, img_syn.shape[0])
                        grayscale_cam = []

                        for i in range(0, img_syn.shape[0], batch_size_cam):
                            batch_img = img_syn[i:i + batch_size_cam]
                            batch_img = ensure_float32(batch_img, args.device)

                            # 修复1: 使用类索引列表作为targets（更简单可靠）
                            # 为batch中的每个图像指定目标类别c
                            targets = [c] * batch_size_cam

                            # 修复2: 确保输入是float32，并且模型也是float32
                            with torch.no_grad():
                                # 生成CAM（使用no_grad避免影响主训练的梯度）
                                cam_batch = cam(input_tensor=batch_img, targets=targets)

                            grayscale_cam.append(cam_batch)

                        grayscale_cam = np.concatenate(grayscale_cam, axis=0)
                        grayscale_cam = 1 - grayscale_cam  # 反转CAM，突出重要区域
                        cam_list.append(grayscale_cam)

                    except Exception as e:
                        cam_error_count += 1
                        logger.error(f"Error generating CAM for class {c} (iter {it}): {str(e)}")
                        # 打印详细的类型信息用于调试
                        logger.error(f"  - img_syn dtype: {img_syn.dtype}")
                        logger.error(f"  - img_syn device: {img_syn.device}")
                        logger.error(f"  - model dtype: {next(net_cam.parameters()).dtype}")
                        logger.error(f"  - model device: {next(net_cam.parameters()).device}")

                        grayscale_cam = np.ones((img_syn.shape[0], im_size[0], im_size[1]))
                        cam_list.append(grayscale_cam)

                    # 计算embed输出
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                optimizer_img.zero_grad()
                loss.backward()

                ''' Apply CAM mask to gradient '''
                try:
                    cam_tensor = torch.tensor(np.concatenate(cam_list, axis=0),
                                              dtype=torch.float32, device=args.device).unsqueeze(1)
                    if image_syn.grad is not None:
                        # 确保梯度也是float32
                        image_syn.grad = image_syn.grad.float()
                        with torch.no_grad():
                            image_syn.grad = image_syn.grad * cam_tensor
                    else:
                        logger.warning("No gradient for image_syn at iteration %d" % it)
                except Exception as e:
                    logger.error(f"Error applying CAM mask: {str(e)}")

                ''' Wavelet domain update '''
                with torch.no_grad():
                    for i in range(image_syn.shape[0]):
                        img = image_syn[i].unsqueeze(0)
                        img = ensure_float32(img, args.device)

                        if image_syn.grad is not None:
                            grad = image_syn.grad[i].unsqueeze(0)
                            grad = ensure_float32(grad, args.device)

                            img_low, img_high = DWT(img)
                            grad_low, grad_high = DWT(grad)

                            # 确保小波变换后的张量也是float32
                            img_low = img_low.float()
                            grad_low = grad_low.float()

                            updated_low = img_low - args.lr_w * grad_low
                            updated_img = IDWT((updated_low, img_high))
                            updated_img = ensure_float32(updated_img, args.device)
                            image_syn[i].copy_(updated_img.squeeze(0))

                optimizer_img.step()
                loss_avg = loss.item() / num_classes

            if it % 10 == 0:
                logger.info('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            # 保存最终结果
            if it == args.Iteration:
                logger.info('=' * 50)
                logger.info('Training completed! Saving results...')
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                save_file = os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_%s_%s_exp%d.pt' % (
                    args.method, args.dataset, args.model, args.ipc,
                    args.proxy_model, args.cam_type, exp))
                torch.save({
                    'data': data_save,
                    'accs_all_exps': accs_all_exps,
                    'cam_proxy': net_cam.state_dict(),
                    'args': args.__dict__
                }, save_file)
                logger.info('Results saved to: {}'.format(save_file))
                logger.info('=' * 50)

    # 打印最终结果
    logger.info('\n' + '=' * 60)
    logger.info('==================== Final Results ====================')
    logger.info('=' * 60)
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s' % (args.num_exp, args.model))
        logger.info('Evaluate %d random %s models' % (len(accs), key))
        logger.info('Final Accuracy - Mean = %.2f%%, Std = %.2f%%' % (np.mean(accs) * 100, np.std(accs) * 100))
        logger.info('=' * 60)


if __name__ == '__main__':
    main()
