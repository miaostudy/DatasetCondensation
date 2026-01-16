import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
# from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from utils import train_CAM, get_loops, get_dataset, get_dataset_med, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, wavelet_transform, inverse_wavelet_transform, get_optimizer
from pytorch_wavelets import DWTForward, DWTInverse

# from tensorboardX import SummaryWriter
import logging
# os.environ['CUDA_VISIBLE_DEVICES']='0'
def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger

# os.environ['CUDA_VISIBLE_DEVICES']='7'
class_names = ['MI', 'CY', 'EP', 'HSIL', 'CC']

def plot_and_save_confusion_matrix(cm, class_names, save_path, args, exp):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # 保存图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, args.method+'_'+str(exp)+'_confusion_matrix.png'))
    plt.close()

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--log_dir',type=str,default='./log_0905' )
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--eval_mode', type=str, default='M', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    # parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    # parser.add_argument('--epoch_eval_train', type=int, default=1, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    # parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--lr_proxy', type=float, default=1e-4, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_w',type=float,default=0.1)
    parser.add_argument('--batch_real', type=int, default=256 ,help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result_0905', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--proxy_model', type=str, default='ResNet18', help='proxy model for uncertainty estimation')
    parser.add_argument('--epoch_proxy', type=int, default=5, help='epochs for training the CAM proxy')
    # parser.add_argument('--epoch_proxy', type=int, default=1, help='epochs for training the CAM proxy')
    parser.add_argument('--save_model_path', type=str, default='./Model_Saved_new', help='proxy model for uncertainty estimation')
    parser.add_argument('--validation_step',type = int ,default= 1,help='validation interval' )
    parser.add_argument('--temperature', type=float, default=0.5, help='validation interval')
    parser.add_argument('--mode',type=str,default='CAMDM')
    parser.add_argument('--cam_path', type=str, default='./CAM_dir', help='dataset path')
    parser.add_argument('--cam_type', type=str, default='GradCAM', help='dataset path')


    args = parser.parse_args()
    args.method = 'WCAMDM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    # log_dir = args.log_dirs
    # writer = SummaryWriter(log_dir=log_dir)

    root = os.getcwd()
    # setup logger
    log_name = args.mode+'_IPC_{}_Model_{}_Proxy_{}_Epoch_{}_Eval_{}_CAM_{}_lrw_{}_limg_{}_Data_{}'.format(args.ipc,args.model,args.proxy_model,args.epoch_proxy,args.eval_mode, args.cam_type, args.lr_w, args.lr_img,args.dataset)
    log_path = os.path.join(args.log_dir,log_name)

    logger = build_logger('./log_1029', cfgname=log_name)
    args.logger = logger

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.cam_path):
        os.mkdir(args.cam_path)



    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    logger.info('eval_it_pool: {}'.format(eval_it_pool))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_dataset_med(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):

        logger.info('\n================== Exp %d ==================\n ' % exp)
        logger.info('Hyper-parameters: \n{}'.format(args.__dict__))
        logger.info('Evaluation model pool: {}'.format(model_eval_pool))
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(torch.tensor(dst_train[i][0]), dim=0) for i in range(len(dst_train))]
        # images_all = [torch.unsqueeze(torch.tensor(item), dim=0) for item in dst_train.imgs]
        # images_all = [item.unsqueeze(dim=-1) for item in images_all]
        # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        labels_all = [torch.tensor(dst_train[i][1]) for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        images_all = images_all.permute(0, 3, 1, 2)



        for c in range(num_classes):
            logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c

            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            logger.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            logger.info('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        logger.info('%s training begins'%get_time())

        ''' Train CAM proxy'''
        net_cam = get_network(args.proxy_model, channel, num_classes, im_size).to(args.device)  # get a random model
        # # 检查并访问原始模型的features
        # if isinstance(net_cam, torch.nn.DataParallel):
        #     model = net_cam.module
        net_cam.train()
        train_CAM(model=net_cam, trainloader=trainloader, args=args)
        # 选择目标层（通常是最后一个卷积层）
        if args.proxy_model == 'ConvNet':
            target_layer = net_cam.features[-1]  # 最后一个卷积层之前的激活层
        elif args.proxy_model == 'ResNet18':
            target_layer = net_cam.layer4[-1]
        # 初始化 GradCAM，移除 use_cuda 参数
        if args.cam_type == 'GradCAM':
            cam = GradCAM(model=net_cam, target_layers=[target_layer])
        elif args.cam_type == 'GradCAM++':
            cam = GradCAMPlusPlus(model=net_cam, target_layers=[target_layer])
        elif args.cam_type == 'XGradCAM':
            cam = XGradCAM(model=net_cam, target_layers=[target_layer])
        for it in range(args.Iteration+1):
            if it%1000==0:
                logger.info('Epoch: {}'.format(it))
            ''' Evaluate synthetic data '''
            save_path_cms = './cms_'+args.method
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    logger.info('DSA augmentation strategy: \n{}'.format(args.dsa_strategy))
                    logger.info('DSA augmentation parameters: \n{}'.format(args.dsa_param.__dict__))

                    accs = []
                    sens = []
                    spes = []
                    f1s = []
                    cms = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        # _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        _, acc_train, acc_test, sensitivity, specificity, f1,cm  = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        sens.append(sensitivity)
                        spes.append(specificity)
                        f1s.append(f1)
                        cms.append(cm)
                    logger.info('Evaluate %d random %s, ACCmean = %.4f ACCstd = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    logger.info('Evaluate %d random %s, SENmean = %.4f SENstd = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(sens), np.std(sens)))
                    logger.info('Evaluate %d random %s, SPEmean = %.4f SPEstd = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(spes), np.std(spes)))
                    logger.info('Evaluate %d random %s, F!mean = %.4f F!std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(f1s), np.std(f1s)))

                    if it == eval_it_pool[-1]:
                        combined_cm =  np.sum(cms, axis=0)
                        plot_and_save_confusion_matrix(combined_cm, class_names, save_path_cms, args,exp)


                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train'''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            # train_CAM(model=net, trainloader=trainloader, args=args)
            # # 选择目标层（通常是最后一个卷积层）
            # target_layer = net.features[-1]  # 最后一个卷积层之前的激活层
            # # 初始化 GradCAM，移除 use_cuda 参数
            # cam = GradCAM(model=net, target_layers=[target_layer])

            '''Freeze Net and optimize images'''
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0
            # 定义小波变换和逆小波变换的模块
            DWT = DWTForward(J=1, wave='haar').to(image_syn.device)  # J表示分解的层数
            IDWT = DWTInverse(wave='haar').to(image_syn.device)
            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                cam_list = []
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)


                    ''' Gen CAM for img_syn'''
                    # 生成 Grad-CAM 热力图（会为批次中的每个图像生成一个热力图）
                    grayscale_cam = 1-cam(input_tensor=img_syn.float())
                    # grayscale_cam_ori = cam(input_tensor=img_syn)
                    cam_list.append(grayscale_cam)
                    # 遍历批次中的每张图像，显示或处理热力图
                    # for i in range(len(img_syn)):
                    # if it % 1000 == 0 and c ==2:
                    #     logger.info('VIS_Epoch: {}'.format(it))
                    #     grayscale_cam_image = grayscale_cam_ori[0, :]
                    #     img_syn_vis = copy.deepcopy(img_syn.detach().cpu())
                    #     plt.imshow(img_syn_vis[0].permute(1, 2, 0).cpu().numpy())
                    #     plt.axis('off')
                    #     # plt.title("image epoch:{}".format(it))
                    #     save_path = os.path.join(args.cam_path,"image epoch:{}.png".format(it))
                    #     plt.savefig(save_path)
                    #     plt.close()
                    #     for ch in range(channel):
                    #         img_syn_vis[:, ch] = img_syn_vis[:, ch] * std[ch] + mean[ch]
                    #     img_syn_vis[img_syn_vis < 0] = 0.0
                    #     img_syn_vis[img_syn_vis > 1] = 1.0
                    #     # 获取批次中第 i 张图像的热力图
                    #     visualization = show_cam_on_image(img_syn_vis[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam_image, use_rgb=True)
                    #
                    #     plt.imshow(visualization)
                    #     plt.axis('off')
                    #     # plt.title("Grad-CAM for image epoch:{}".format(it))
                    #     save_path = os.path.join(args.cam_path, "Grad-CAM for image epoch:{}.png".format(it))
                    #     plt.savefig(save_path)
                    #     # plt.show()
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    #############################################################################

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            cam_tensor = torch.tensor(np.concatenate(cam_list, axis=0)).unsqueeze(1).to(args.device)
            with torch.no_grad():
                image_syn.grad = image_syn.grad * cam_tensor

            #################################################
            for i in range(image_syn.shape[0]):
                img = image_syn[i].unsqueeze(0) # 需要将图像增加维度 (B, C, H, W)
                grad = image_syn.grad[i].unsqueeze(0)

                # 对image_syn和其梯度进行小波变换
                img_low, img_high = DWT(img)  # 低频和高频分量
                grad_low, grad_high = DWT(grad)  # 梯度的低频和高频分量

                # 在小波域上手动更新：使用grad_wavelet直接调整img_wavelet的分量
                # 更新低频分量
                updated_low = img_low - args.lr_w * grad_low

                # # 更新高频分量
                # updated_high = []
                # for img_h, grad_h in zip(img_high, grad_high):
                #     updated_high.append(img_h - args.lr_w * grad_h)

                # 逆小波变换，获得更新后的image_syn
                # updated_img = IDWT((updated_low, updated_high))  # 逆变换
                updated_img = IDWT((updated_low, img_high))  # 逆变换
                with torch.no_grad():
                    image_syn[i].copy_(updated_img.squeeze(0))

            # ##################################################
            optimizer_img.step()
            loss_avg += loss.item()
            loss_avg /= (num_classes)
            # print('1 Round OPT Finished')
            # logger.info('Loss: {}'.format(loss_avg))
            if it%10 == 0:
                logging.info('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, 'cam_proxy': net_cam.state_dict()}, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_%s_%s_%d.pt'%(args.method, args.dataset, args.model, args.ipc,args.proxy_model,args.cam_type,args.epoch_proxy)))


    logging.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


