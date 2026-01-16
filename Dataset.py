import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from medmnist import  PathMNIST, OCTMNIST, ChestMNIST, BreastMNIST, TissueMNIST, BloodMNIST, PneumoniaMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST
import pickle
from PIL import Image
import pywt


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


train_path = '/mnt/lyp/whole_oct_train.pkl'
test_path = '/mnt/lyp/whole_oct_test.pkl'
# test_path = '/mnt/lyp/huaxi_oct_test.pkl'
# test_path = '/mnt/lyp/xiangya_oct_test.pkl'

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

def get_dataset_med(dataset, data_path):
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

        # TinyImageNet的标准均值方差

        mean = [0.485, 0.456, 0.406]

        std = [0.229, 0.224, 0.225]

        # 定义transform（Resize到64x64，与im_size一致）

        transform = transforms.Compose([

            transforms.Resize((64, 64)),

            transforms.ToTensor(),

            transforms.Normalize(mean=mean, std=std)

        ])

        # 1. 加载训练集（train文件夹下每个类一个子文件夹）

        train_dir = os.path.join(data_path, 'train')

        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

        class_to_idx = train_dataset.class_to_idx  # wnid到类索引的映射

        wnids = list(class_to_idx.keys())  # 200个类的wnid

        # 2. 加载验证集（需要处理标注文件）

        val_dir = os.path.join(data_path, 'val')

        annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        val_dataset = TinyImageNetValDataset(

            val_dir=val_dir,

            annotations_file=annotations_file,

            class_to_idx=class_to_idx,

            transform=transform

        )

        # 3. 加载测试集（test文件夹下只有images子文件夹，无标注，这里用验证集作为测试集）

        # 注意：TinyImageNet官方测试集没有公开标注，通常用验证集作为测试集

        dst_train = train_dataset

        dst_test = val_dataset

        # 4. 加载类名（从words.txt映射wnid到人类可读名称）

        words_file = os.path.join(data_path, 'words.txt')

        wnid_to_name = {}

        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')

                wnid = parts[0]

                name = parts[1].split(',')[0]  # 取第一个名称（避免过长）

                wnid_to_name[wnid] = name

        # 生成类名列表（与train_dataset的class_to_idx顺序一致）

        class_names = [wnid_to_name[wnid] for wnid in wnids]

        print(f"Loaded TinyImageNet: {len(dst_train)} train images, {len(dst_test)} test images")
        print(f"Number of classes: {num_classes}")

    elif dataset == 'oct':
        channel = 3
        im_size = (64,64)
        num_classes = 5
        mean = [0.3124, 0.3124, 0.3124]
        std =  [0.2206, 0.2206, 0.2206]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = OCTDataset(img_size=im_size[0], data_list=train_path)
        dst_test = OCTDataset(img_size=im_size[0], data_list=test_path)
        class_names = ['mild inflammation','cyst','ectropion','high-grade squamous intraepithelial lesion','cervical cancer']
    elif dataset =='PathMNIST':
        channel = 3
        im_size = (28,28)
        num_classes = 9
        mean = [0.4823, 0.4823, 0.4823]
        std = [0.2309, 0.2309, 0.2309]
        dst_train_raw = PathMNIST(split='train',download=True)
        dst_test_raw = PathMNIST(split='test',download=True)
        train_imgs = dst_train_raw.imgs/255.0
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        # dst_train = TensorDataset(train_imgs,train_labels)
        test_imgs = dst_test_raw.imgs/255.0
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0,3,1,2), test_labels))
        # class_names = [str(i) for i in range(num_classes)]
        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))

    elif dataset == 'OCTMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 4
        mean = [0.4850]
        std = [0.2292]
        dst_train_raw = OCTMNIST(split='train', download=True)

        dst_test_raw = OCTMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs/255.0,axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs,train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs/255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0,3,1,2), test_labels))
        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'ChestMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 2
        mean = [0.6101]
        std = [0.2075]
        dst_train_raw = ChestMNIST(split='train', download=True)

        dst_test_raw = ChestMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs/255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs/255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0,3,1,2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'BreastMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 2
        mean = [0.1751]
        std = [0.3048]
        dst_train_raw = BreastMNIST(split='train', download=True)

        dst_test_raw = BreastMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'BloodMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 8
        mean = [0.7569]
        std = [0.2054]
        dst_train_raw = BloodMNIST(split='train', download=True)

        dst_test_raw = BloodMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'TissueMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 8
        mean = [0.5]
        std = [0.5]
        dst_train_raw = TissueMNIST(split='train', download=True)

        dst_test_raw = TissueMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'PneumoniaMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 2
        mean = [0.4737]
        std = [0.2511]
        dst_train_raw = PneumoniaMNIST(split='train', download=True)

        dst_test_raw = PneumoniaMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'OrganAMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.1220]
        std = [0.3023]
        dst_train_raw = OrganAMNIST(split='train', download=True)

        dst_test_raw = OrganAMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'OrganCMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.5581]
        std = [0.2413]
        dst_train_raw = OrganCMNIST(split='train', download=True)

        dst_test_raw = OrganCMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))
    elif dataset == 'OrganSMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.0762]
        std = [0.2054]
        dst_train_raw = OrganSMNIST(split='train', download=True)

        dst_test_raw = OrganSMNIST(split='test', download=True)

        train_imgs = np.expand_dims(dst_train_raw.imgs / 255.0, axis=-1)
        train_labels = [item[0] for item in dst_train_raw.labels]
        dst_train = list(zip(train_imgs, train_labels))
        test_imgs = np.expand_dims(dst_test_raw.imgs / 255.0, axis=-1)
        test_labels = [item[0] for item in dst_test_raw.labels]
        dst_test = list(zip(test_imgs.transpose(0, 3, 1, 2), test_labels))

        class_names = [str(i) for i in range(num_classes)]
        print('Loaded the dataset:{}'.format(dataset))

    else:
        exit('unknown dataset: %s'%dataset)



    # ######### randomly select train and valid set with 8:2 ########
    # # random.shuffle(dst_train)
    # split_point = int(len(dst_train) * 0.8)
    # train_selected = Subset(dst_train,range(split_point))
    # valid_selected = Subset(dst_train,range(split_point,len(dst_train)))
    # ######################################################################

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    # valid_loader = torch.utils.data.DataLoader(valid_selected, batch_size=256, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=0)
    # train_proxy_loader = torch.utils.data.DataLoader(train_selected, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader
