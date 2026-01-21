import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, f1_score
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, ViTTinyWithEmbedding

from medmnist import  PathMNIST, OCTMNIST, ChestMNIST, BreastMNIST, TissueMNIST, BloodMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, RetinaMNIST, DermaMNIST
import pickle
from PIL import Image
import pywt
import torchvision.models as models
import pandas as pd

# 小波变换和逆小波变换的辅助函数
def wavelet_transform(image, wavelet='haar', mode='symmetric'):
    coeffs = pywt.wavedec2(image.detach().cpu().numpy(), wavelet, mode=mode)
    return [torch.tensor(c, dtype=torch.float32).to(image.device) for c in coeffs]

def inverse_wavelet_transform(coeffs, wavelet='haar', mode='symmetric'):
    image_rec = pywt.waverec2([c.detach().cpu().numpy() for c in coeffs], wavelet, mode=mode)
    return torch.tensor(image_rec, dtype=torch.float32).to(coeffs[0].device)

# 定义一个优化器用于更新小波系数
def get_optimizer(params, lr=0.01):
    return torch.optim.SGD(params, lr=lr)


def _load_medmnist_as_tensor(data_raw, channel):
    # 1. 归一化并转为 Tensor
    imgs = torch.tensor(data_raw.imgs).float() / 255.0

    # 2. 调整维度 (N, H, W) 或 (N, H, W, C) -> (N, C, H, W)
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    elif imgs.ndim == 4:
        imgs = imgs.permute(0, 3, 1, 2)  # (N, H, W, 3) -> (N, 3, H, W)

    # 3. 处理标签
    labels = torch.tensor([item[0] for item in data_raw.labels]).long()

    return imgs, labels

train_path = '/mnt/lyp/whole_oct_train.pkl'
test_path = '/mnt/lyp/whole_oct_test.pkl'
# test_path = '/mnt/lyp/huaxi_oct_test.pkl'
# test_path = '/mnt/lyp/xiangya_oct_test.pkl'
class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, annotations_file, class_to_idx, transform=None):
        self.val_dir = val_dir
        self.annotations = pd.read_csv(annotations_file, sep='\t', header=None,
                                       names=['filename', 'wnid', 'x1', 'y1', 'x2', 'y2'])
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.val_dir, 'images', row['filename'])
        img = Image.open(img_path).convert('RGB')

        # 裁剪边界框（TinyImageNet验证集提供了精确边界框）
        img = img.crop((row['x1'], row['y1'], row['x2'], row['y2']))

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[row['wnid']]
        return img, label


class OCTDataset(Dataset):
    def __init__(self, img_size, data_list):
        self.trainsize = (img_size, img_size)
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        self.transform_center = transforms.Compose([

            transforms.Resize(self.trainsize),

            transforms.ToTensor(),
            transforms.Normalize([0.3124, 0.3124, 0.3124], [0.2206, 0.2206, 0.2206])
        ])

        # self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]



        img_path = data_pac['img_root']
        # cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        return img_torch, label

    def __len__(self):
        return self.size

# 自定义TensorDataset（如果utils中没有定义）
class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


def _load_medmnist_as_tensor(data_raw, channel):
    """
    辅助函数：将 MedMNIST 的 numpy 数据转换为 (N, C, H, W) 的 PyTorch Tensor
    """
    # 1. 归一化并转为 Float Tensor
    imgs = torch.tensor(data_raw.imgs).float() / 255.0

    # 2. 调整维度
    # MedMNIST 原始格式:
    # 灰度图: (N, 28, 28) -> 需要变为 (N, 1, 28, 28)
    # 彩色图: (N, 28, 28, 3) -> 需要变为 (N, 3, 28, 28)
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)
    elif imgs.ndim == 4:
        imgs = imgs.permute(0, 3, 1, 2)

    # 3. 处理标签
    labels = torch.tensor([item[0] for item in data_raw.labels]).long()

    return imgs, labels


def get_dataset_med(dataset, data_path):
    # ================== 1. MedMNIST 系列数据集配置 ==================
    # 格式: {数据集名: {通道数, 类别数, 均值, 方差}}
    # 均值和方差沿用了你原始代码中的设定，Retina/Derma 使用 0.5
    medmnist_config = {
        'PathMNIST': {'channel': 3, 'cls': 9, 'mean': [0.4823] * 3, 'std': [0.2309] * 3},
        'OCTMNIST': {'channel': 1, 'cls': 4, 'mean': [0.4850], 'std': [0.2292]},
        'ChestMNIST': {'channel': 1, 'cls': 2, 'mean': [0.6101], 'std': [0.2075]},
        'BreastMNIST': {'channel': 1, 'cls': 2, 'mean': [0.1751], 'std': [0.3048]},
        'TissueMNIST': {'channel': 1, 'cls': 8, 'mean': [0.5], 'std': [0.5]},
        'BloodMNIST': {'channel': 3, 'cls': 8, 'mean': [0.7569], 'std': [0.2054]},
        'PneumoniaMNIST': {'channel': 1, 'cls': 2, 'mean': [0.4737], 'std': [0.2511]},
        'OrganAMNIST': {'channel': 1, 'cls': 11, 'mean': [0.1220], 'std': [0.3023]},
        'OrganCMNIST': {'channel': 1, 'cls': 11, 'mean': [0.5581], 'std': [0.2413]},
        'OrganSMNIST': {'channel': 1, 'cls': 11, 'mean': [0.0762], 'std': [0.2054]},
        'RetinaMNIST': {'channel': 3, 'cls': 5, 'mean': [0.5] * 3, 'std': [0.5] * 3},
        'DermaMNIST': {'channel': 3, 'cls': 7, 'mean': [0.5] * 3, 'std': [0.5] * 3}
    }

    if dataset in medmnist_config:
        cfg = medmnist_config[dataset]
        channel = cfg['channel']
        im_size = (28, 28)
        num_classes = cfg['cls']
        mean = cfg['mean']
        std = cfg['std']
        class_names = [str(i) for i in range(num_classes)]

        # 动态导入 medmnist
        import medmnist
        # 获取对应的数据集类，例如 medmnist.PathMNIST
        DataClass = getattr(medmnist, dataset)

        # 加载原始数据
        dst_train_raw = DataClass(split='train', download=True, root=data_path)
        dst_test_raw = DataClass(split='test', download=True, root=data_path)

        # === 核心转换：转为 Tensor (N, C, H, W) ===
        train_imgs, train_labels = _load_medmnist_as_tensor(dst_train_raw, channel)
        test_imgs, test_labels = _load_medmnist_as_tensor(dst_test_raw, channel)

        # 封装为 TensorDataset
        dst_train = TensorDataset(train_imgs, train_labels)
        dst_test = TensorDataset(test_imgs, test_labels)

        print(f'Loaded MedMNIST dataset: {dataset} (Train: {len(dst_train)}, Test: {len(dst_test)})')

    # ================== 2. 通用/自然图像数据集 ==================
    elif dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # 假设 data_path 下有 train 和 val 目录
        train_dir = os.path.join(data_path, 'train')
        dst_train = datasets.ImageFolder(root=train_dir, transform=transform)

        val_dir = os.path.join(data_path, 'val')
        dst_test = datasets.ImageFolder(root=val_dir, transform=transform)

        num_classes = len(dst_train.classes)
        # 尝试读取类别名映射
        words_file = os.path.join(data_path, 'words.txt')
        wnid_to_name = {}
        if os.path.exists(words_file):
            with open(words_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        wnid_to_name[parts[0]] = parts[1].split(',')[0]

        class_names = [wnid_to_name.get(wnid, wnid) for wnid in dst_train.classes]
        print(f"Loaded TinyImageNet: {len(dst_train)} train, {len(dst_test)} test, {num_classes} classes")

    elif dataset == 'oct':
        channel = 3
        im_size = (64, 64)
        num_classes = 5
        mean = [0.3124, 0.3124, 0.3124]
        std = [0.2206, 0.2206, 0.2206]

        dst_train = OCTDataset(img_size=im_size[0], data_list=train_path)
        dst_test = OCTDataset(img_size=im_size[0], data_list=test_path)
        class_names = ['mild inflammation', 'cyst', 'ectropion', 'high-grade squamous intraepithelial lesion',
                       'cervical cancer']

    else:
        exit('unknown dataset: %s' % dataset)

    # 构造 DataLoader
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=0)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader
def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)
    # Add ViT-Tiny
    elif model == 'ViT-Tiny':
        # net = models.vit_tiny_patch16_224(num_classes=num_classes)
        net = ViTTinyWithEmbedding(num_classes=num_classes, image_size = im_size[0])

    # Add ConvNext-Tiny
    # elif model == 'ConvNext-Tiny':
    #     # net = models.convnext_tiny(num_classes=num_classes)
    #     net = ConvNextTinyWithEmbedding(num_classes=num_classes)
    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img.float())
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch('train', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

    # Calculate F1, Sensitivity, Specificity, Confusion Matrix
    net.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            input = input.to(args.device)
            target = target.to(args.device)
            output = net(input)
            _, pred = torch.max(output, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    cm = confusion_matrix(targets, preds)
    # Handle multi-class case for Sensitivity/Specificity (using macro avg)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    with np.errstate(divide='ignore', invalid='ignore'):
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
    sen = np.nanmean(sen)
    spe = np.nanmean(spe)
    f1 = f1_score(targets, preds, average='macro')

    print(
        '%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f, test_sen =%.4f, test_spe =%.4f, test_f1 =%.4f' % (
            get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test, sen, spe, f1))

    # FIX: 必须返回 7 个值以匹配 mWCAMDM.py
    return loss_train, acc_train, acc_test, sen, spe, f1, cm


def epoch_full(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    all_preds = []
    all_labels = []

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        preds = torch.argmax(output, dim=1).cpu().numpy()

        # 累积预测结果和真实标签
        all_preds.extend(preds)
        all_labels.extend(lab.cpu().numpy())

        acc = np.sum(np.equal(preds, lab.cpu().numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    # 计算灵敏度、特异性、F1-score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # # 计算每个类别的灵敏度（Recall）
    # sensitivity = recall_score(all_labels, all_preds, average=None)  # 每个类别的灵敏度
    # 计算每个类别的灵敏度
    sensitivities = []

    for i in range(len(cm)):
        tp = cm[i, i]  # True Positives (TP)
        fn = np.sum(cm[i, :]) - tp  # False Negatives (FN)
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        sensitivities.append(sensitivity)
    # 计算每个类别的特异性
    specificity = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        tn = tn if tn > 0 else 1  # 避免除以零
        specificity.append(tn / (tn + fp))
    specificity = np.array(specificity)  # 每个类别的特异性

    # 计算F1-score
    f1 = f1_score(all_labels, all_preds, average='macro')  # 宏平均F1-score
    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificity)
    return loss_avg, acc_avg, sensitivity, specificity, f1, cm

# def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
#     net = net.to(args.device)
#     images_train = images_train.to(args.device)
#     labels_train = labels_train.to(args.device)
#     lr = float(args.lr_net)
#     Epoch = int(args.epoch_eval_train)
#     lr_schedule = [Epoch//2+1]
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#     criterion = nn.CrossEntropyLoss().to(args.device)
#
#     dst_train = TensorDataset(images_train, labels_train)
#     trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
#
#     start = time.time()
#     for ep in range(Epoch+1):
#         loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
#         if ep in lr_schedule:
#             lr *= 0.1
#             optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#
#     time_train = time.time() - start
#     loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
#     print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
#
#     return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' # 'S' or 'M'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.batchmode = False
        self.latestseed = -1  # <--- CRITICAL FIX: 补充缺失的属性

    def set_mode(self, mode):
        self.aug_mode = mode

def set_seed_DiffAug(param):
    if param.batchmode:
        torch.manual_seed(0)
def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy is None:
        return x

    if seed == -1:
        param.batchmode = False
    elif seed == 0:
        param.batchmode = True  # use a fixed random seed
    else:
        param.batchmode = True  # use a fixed random seed with specific value

    if param.batchmode and seed != 0:
        torch.manual_seed(seed)

    AUGMENT_FNS = {
        'color': [rand_brightness, rand_saturation, rand_contrast],
        'crop': [rand_crop],
        'cutout': [rand_cutout],
        'flip': [rand_flip],
        'scale': [rand_scale],
        'rotate': [rand_rotate],
    }

    if strategy:
        if param.aug_mode == 'M': # single
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S': # multiple
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x
# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0], [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone() # FIX
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x

def rand_rotate(x, param): # rotate
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0], [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone() # FIX
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x

def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0].clone() # FIX: Added .clone()
    return torch.where(randf < prob, x.flip(3), x)

def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0].clone() # FIX
    return x + (randb - 0.5)*ratio

def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0].clone() # FIX
    return (x - x_mean) * (rands * ratio) + x_mean

def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0].clone() # FIX
    return (x - x_mean) * (randc + 0.1) * ratio + x_mean


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0].clone() # FIX
        translation_y[:] = translation_y[0].clone() # FIX
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0].clone() # FIX
        offset_y[:] = offset_y[0].clone() # FIX
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, 0, x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def load_or_train_model(model, args, train_loader):
    # 检查模型参数文件是否存在
    if os.path.exists(args.checkpoint_path):
        print(f"Found checkpoint at {args.checkpoint_path}, loading parameters...")
        # 读取已训练的网络参数
        model_ckpt = torch.load(args.checkpoint_path)
        for key in list(model_ckpt.keys()):
            new_key = key.replace('module.','')
            model_ckpt[new_key]=model_ckpt.pop(key)
        # new_state_dict = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(model_ckpt)
        print("Model parameters loaded successfully!")
        return model
    else:
        print(f"No checkpoint found at {args.checkpoint_path}, training model...")
        # 如果没有找到预训练的模型，执行训练
        train_CAM(model, train_loader, args)
        return model
        # 训练完毕后保存模型参数
        # torch.save(model.state_dict(), args.checkpoint_path)



def train_CAM(model, trainloader, args):
    num_epochs = args.epoch_proxy
    # 设置设备
    device = args.device
    for param in list(model.parameters()):
        param.requires_grad = True
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('Training CAM')
    for epoch in range(num_epochs):  # 简单训练2个epoch以节省时间
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            # outputs = model(inputs.permute(0,3,1,2).float())
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 打印统计信息
            running_loss += loss.item()
            if i % 100 == 4:  # 每5批次打印一次
                # print(f'[Epoch_CAM: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Model trained and saved to {args.checkpoint_path}!")
    return model



AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
