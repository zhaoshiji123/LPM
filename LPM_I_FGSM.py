import os
import time
# from sqlalchemy import false
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from LPM_attack import attack
from data import norm
from utils import input_diversity, gaussian_kernel, TI_kernel
from model import load_models, get_logits
from data import MyDataset, save_imgs, imgnormalize
from PIL import Image
from loader import ImageNet
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# model_names = ['resnet50','resnet101','resnet152','inception_v3','vgg16','densenet161','adv_inception_v3']
# source_names = ['inception_v4','resnet101']#white +v3+v4+inc-res-v2 gray 1:densnet121\vgg16 2
source_names = ['inception_v3']
simulated_names = ['resnet50','vgg16','densenet161']
target_names_1 = ['inception_resnet_v2']
target_names_2 = ['ens_adv_inception_resnet_v2']
target_names_3 = ['adv_inception_v3']
target_names_4 = ['inception_v4']

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

device = torch.device("cuda:0")
args2 = ""
def parse_arguments():
    parser = argparse.ArgumentParser(description='Transfer attack')
    parser.add_argument('--source_models', nargs="+", default=['resnet101'], help='source models')
    parser.add_argument('--transfer_models', nargs="+", default=['inception_v3'], help='transfer models in black-box')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--TI', type=bool, default=False)
    parser.add_argument('--MI', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=8, help='Number of iterations')
    parser.add_argument('--M', type=int, default=1, help='Number of M')
    parser.add_argument('--path_size', type=int, default=16, help='Patch size in mask')
    parser.add_argument('--rate', type=float, default=0.1, help='image cover rate through mask')
    parser.add_argument('--alpha', type=eval, default=2.0/255., help='Step size')
    parser.add_argument('--epsilon', type=float, default=16, help='The maximum pixel value can be changed')
    parser.add_argument('--input_diversity', type=eval, default="True", help='Whether to use Input Diversity')
    parser.add_argument('--input_path', type=str, default='../input_dir', help='Path of input')
    parser.add_argument('--label_file', type=str, default='dev-1000-new.csv', help='Label file name')
    parser.add_argument('--result_path', type=str, default='main_5_nips_dataset_1000_zsj_version4_30_2/', help='Path of adv images to be saved')
    args = parser.parse_args()
    return args



def single_attack(img,labels,model):
    # X_ori = torch.stack([img])
    X_ori = img.clone()
    X = img
    # labels = torch.stack([label])
    # X = X.type(torch.float32)
    X.to(device)
    labels.to(device)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    for t in range(10):
        X_adv = X + delta
        ensemble_logits = get_logits(X_adv, model)
        loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
        loss.backward()
        grad = delta.grad.clone()
        # Ti
        # grad = F.conv2d(grad, TI_kernel(kernel_size=5, nsig=3), bias=None, stride=1, padding=(2,2), groups=3)
                # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        # Mi
        # grad = grad/torch.mean(torch.abs(grad), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
        # grad_momentum = grad
        # g = grad*mask
        delta.grad.zero_()

        delta.data = delta.data - 1.6/255. * torch.sign(grad)
        delta.data = delta.data.clamp(-16.0/255., 16.0/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv


def save_imgs(X, adv_img_save_folder, filenames, img_type = ''):
    for i in range(X.shape[0]):
        adv_final = X[i].cpu().detach().numpy()
        adv_final = (adv_final * 255).astype(np.uint8)
        adv_img_save_folder1 = adv_img_save_folder+"/1"+img_type
        adv_img_save_folder2 = adv_img_save_folder+"/2"+img_type
        adv_img_save_folder3 = adv_img_save_folder+"/3"+img_type
        adv_img_save_folder4 = adv_img_save_folder+"/4"+img_type
        if not os.path.exists(adv_img_save_folder1):
            os.makedirs(adv_img_save_folder1)
        if not os.path.exists(adv_img_save_folder2):
            os.makedirs(adv_img_save_folder2)
        if not os.path.exists(adv_img_save_folder3):
            os.makedirs(adv_img_save_folder3)
        if not os.path.exists(adv_img_save_folder4):
            os.makedirs(adv_img_save_folder4)
        file_path1 = os.path.join(adv_img_save_folder1, filenames[i].split('/')[-1])
        file_path2 = os.path.join(adv_img_save_folder2, filenames[i].split('/')[-1])
        file_path3 = os.path.join(adv_img_save_folder3, filenames[i].split('/')[-1])
        file_path4 = os.path.join(adv_img_save_folder4, filenames[i].split('/')[-1])
        adv_x_255 = np.transpose(adv_final, (1, 2, 0))
        im = Image.fromarray(adv_x_255)
        # quality can be affects the robustness of the adversarial images
        im.save(file_path1, quality=99)
        im.save(file_path2, quality=95)
        im.save(file_path3, quality=90)
        im.save(file_path4)


def save_imgs_mask(X, adv_img_save_folder, filenames):
    for i in range(X.shape[0]):
        adv_final = X[i].cpu().detach().numpy()
        adv_final = (adv_final * 255).astype(np.uint8)
        adv_img_save_folder1 = adv_img_save_folder+"/mask"
        if not os.path.exists(adv_img_save_folder1):
            os.makedirs(adv_img_save_folder1)
        file_path1 = os.path.join(adv_img_save_folder1, filenames[i].split('/')[-1])
        adv_x_255 = np.transpose(adv_final, (1, 2, 0))
        im = Image.fromarray(adv_x_255)
        # quality can be affects the robustness of the adversarial images
        im.save(file_path1, quality=99)

def run_attack(args):
    global args2
    args2 = args
    batch_size = 1
    input_folder = os.path.join(args.input_path, 'images-resize/')
    adv_img_save_folder = os.path.join(args.result_path, 'I_FGSM_I_10_1.6/')
    if not os.path.exists(adv_img_save_folder):
        os.makedirs(adv_img_save_folder)
    X = ImageNet("NIPS_dataset/dev_dataset/images", "NIPS_dataset/dev_dataset/dev_dataset.csv", transforms)
    data_loader = DataLoader(X, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    device = torch.device("cuda:0")
    source_models = load_models(source_names, device)  # load model, maybe several models
    simulated_models =  load_models(simulated_names, device)

    seed_num = 0  # set seed
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    print('Start attack......')
    suc_white = 0
    acc = 0
    cnt = 0
    # for i, data in enumerate(data_loader, 0):
    i = 0
    unsuccess_y = []
    for images, images_ID,  gt_cpu in tqdm(data_loader):

        gt = gt_cpu.cuda()
        images = images.cuda()
        filenames = images_ID
        # print(images_ID)
        # assert False
        start_t = time.time()
        X = images
        X = X.to(device)
        labels = gt_cpu.to(device)
        # print(X.shape)
        # print(labels.shape)
        # assert False
        X_adv_esemble_12, y, X_mask = attack(X[0],labels[0],source_models,simulated_models)


        save_imgs(X_adv_esemble_12, adv_img_save_folder, filenames, img_type = '_mask_esemble_12')


      
        i += 1

if __name__ == '__main__':
    args = parse_arguments()
    args.patch_size = 32
    args.rate = 0.05
    run_attack(args)
