import os
import time
from sqlalchemy import false
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from attack import attack
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
target_names_5 = ['resnet101']
target_names_6 = ['resnet152']


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
    parser.add_argument('--result_path', type=str, default='I_FGSM_in_main_4_nips_dataset_1000/', help='Path of adv images to be saved')
    args = parser.parse_args()
    return args

def get_mask(patch_size, batch_size, rate):
    width = int(299 / patch_size)
    height = int(299 / patch_size)
    mask = np.random.rand(batch_size, 3, height, width)
    one = np.ones_like(mask)
    zero = np.zeros_like(mask)
    mask = np.where(mask <= rate, zero, one)
    mask = torch.from_numpy(mask).to(device).float()
    mask = F.interpolate(mask, (299, 299), mode='bilinear', align_corners=False)
    return mask

def mask_trans(x, patch_size, batch_size, rate):
    width = int(229 / patch_size)
    height = int(229 / patch_size)
    mask = np.random.rand(batch_size, 3, height, width)
    one = np.ones_like(mask)
    zero = np.zeros_like(mask)
    mask = np.where(mask <= rate, zero, one)
    mask = torch.from_numpy(mask).to(device).float()
    mask = F.interpolate(mask, (229, 229), mode='bilinear', align_corners=False)
    return mask*x


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
    for t in range(8):
        X_adv = X + delta
        ensemble_logits = get_logits(X_adv, model)
        loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
        loss.backward()
        grad = delta.grad.clone()
        # Ti
        # grad = F.conv2d(grad, TI_kernel(kernel_size=5, nsig=3), bias=None, stride=1, padding=(2,2), groups=3)
        # Mi
        # grad = grad/torch.mean(torch.abs(grad), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
        # grad_momentum = grad
        # g = grad*mask
        delta.grad.zero_()

        delta.data = delta.data - 2.0/255 * torch.sign(grad)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv


def save_imgs(X, adv_img_save_folder, filenames):
    for i in range(X.shape[0]):
        adv_final = X[i].cpu().detach().numpy()
        adv_final = (adv_final * 255).astype(np.uint8)
        adv_img_save_folder1 = adv_img_save_folder+"/1"
        adv_img_save_folder2 = adv_img_save_folder+"/2"
        adv_img_save_folder3 = adv_img_save_folder+"/3"
        adv_img_save_folder4 = adv_img_save_folder+"/4"
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


def run_attack(args):
    global args2
    args2 = args
    batch_size = 1
    input_folder = os.path.join(args.input_path, 'images-resize/')
    adv_img_save_folder = os.path.join(args.result_path, 'I_FGSM/')
    if not os.path.exists(adv_img_save_folder):
        os.makedirs(adv_img_save_folder)
    # csv_path = "input_dir/dev-1000-new.csv"
    # input_folder = "input_dir/images-resize-224/"
    # data_set = MyDataset(csv_path=csv_path, path=input_folder)
    # data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # X = ImageNet("NIPS_dataset/dev_dataset/images", "NIPS_dataset/dev_dataset/dev_dataset.csv", transforms)
    # X = ImageNet("main_5_nips_dataset_1000_zsj_version4_17/I_FGSM_I_10_1.6/1_esemble_mask_2", "NIPS_dataset/dev_dataset/dev_dataset.csv", transforms)
    # X = ImageNet("main_5_nips_dataset_1000_zsj_version1/I_FGSM_I_10_1.6/1", "NIPS_dataset/dev_dataset/dev_dataset.csv", transforms)
    X = ImageNet(r"C:\Users\shiji\国秋师兄工作\masks_trans\FGSM_ESEMBLE_FGSM_DTS\MI_FGSM_DTS_in_main_5_nips_dataset_1000\MI_FGSM_DTS\1", "NIPS_dataset/dev_dataset/dev_dataset.csv", transforms)

    data_loader = DataLoader(X, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    device = torch.device("cuda:0")
    source_models = load_models(source_names, device)  # load model, maybe several models
    target_model_1 =  load_models(target_names_1, device)
    target_model_2 =  load_models(target_names_2, device)
    target_model_3 =  load_models(target_names_3, device)
    target_model_4 =  load_models(target_names_4, device)
    target_model_5 =  load_models(target_names_5, device)
    target_model_6 =  load_models(target_names_6, device)
    # simulated_models =  load_models(simulated_names, device)
    seed_num = 0  # set seed
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    print('Start attack......')
    suc_white = 0
    suc_black_1 = 0
    suc_black_2 = 0
    suc_black_3 = 0
    suc_black_4 = 0
    suc_black_5 = 0
    suc_black_6 = 0
    acc = 0
    cnt = 0
    # for i, data in enumerate(data_loader, 0):
    i = 0
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        # if i == 100:
        #     break
        gt = gt_cpu.cuda()
        images = images.cuda()
        filenames = images_ID
        start_t = time.time()
        X = images

        # X = F.interpolate(X, (224, 224), mode='bilinear', align_corners=False)
        X = X.to(device)
        labels = gt_cpu.to(device)
        out = get_logits(X, source_models)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_white += (labels.shape[0] - torch.sum(correct == 0).item())
        cnt += labels.shape[0]
        print((i+1)*batch_size,"White Box Success Rate: {:.4f} ".format(suc_white))
        out = get_logits(X, target_model_1)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_1 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_1: {:.4f} ".format(suc_black_1))
        out = get_logits(X, target_model_2)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_2 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_2: {:.4f} ".format(suc_black_2))
        out = get_logits(X, target_model_3)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_3 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_3: {:.4f} ".format(suc_black_3))
        out = get_logits(X, target_model_4)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_4 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_4: {:.4f} ".format(suc_black_4))
        out = get_logits(X, target_model_5)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_5 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_5: {:.4f} ".format(suc_black_5))
        out = get_logits(X, target_model_6)
        adv_label = torch.argmax(out,dim=1)
        correct = adv_label - labels
        suc_black_6 += (labels.shape[0] - torch.sum(correct == 0).item())
        print((i+1)*batch_size,"Black Box Success Rate for target_model_6: {:.4f} ".format(suc_black_6))
        i += 1
        
if __name__ == '__main__':
    start_time = time.time()
    args = parse_arguments()
    args.patch_size = 16
    args.rate = 0.05
    run_attack(args)
    print('Total time(seconds):{:.3f}'.format(time.time()-start_time))
