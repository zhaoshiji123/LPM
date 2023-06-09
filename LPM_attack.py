from re import I, subn
from warnings import simplefilter

import numpy as np
from torchvision import transforms 
import torch
from time import *
import sys
import torch.nn as nn
import torch.nn.functional as F
from model import load_models, get_logits
from utils import TI_kernel
from data import norm
from scikit_opt.sko.GA import GA
from scikit_opt.sko.DE import DE
import os
from PIL import Image

trans = transforms.Compose([
                transforms.ToTensor(),
            ])

MAX = 10000
inf = -MAX
suf = MAX
CLUSTER = 10   # mask中每cluster个相邻的位置用一个数表示
patch_size = 32
WIDTH=299
HEIGHT=299
REAR_LEN = WIDTH*HEIGHT%CLUSTER
device = torch.device("cuda:0")



def save_imgs(X):
    # for i in range(X.shape[0]):
    adv_final = X[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    # print(adv_x_255.shape)
    im = Image.fromarray(adv_x_255)
    # quality can be affects the robustness of the adversarial images
    im.save("mask0.png", quality=99)
    assert False



def print_mask(img,mask):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    mask = mask_final[:,None,:,:]
    mask = torch.cat((mask,mask,mask),1)
    mask = mask.float()
    # print(mask.shape)

    mask = mask.cuda()
    X_ori = torch.stack([img])
    X = X_ori

    X[:,:,:288,:288] = X[:,:,:288,:288] * mask[0]
    # save_imgs(X_1)
    
    return X

def batch_attack_final(img,mask,label,white_models):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    mask = mask_final[:,None,:,:]
    mask = torch.cat((mask,mask,mask),1)
    mask = mask.float()
    # print(mask.shape)
    # print(img.shape)
    # mask = F.interpolate(mask, (288, 288), mode='bilinear', align_corners=False)
    mask = mask.cuda()
    
    X_ori = torch.stack([img])
    X = X_ori

    labels = torch.stack([label])
    X.to(device)
    labels.to(device)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    # M_num = int(mask.shape[0]/8)
    cnt = 0
    for t in range(10):
        X_adv = X + delta
        X_adv[:,:,:288,:288] = X_adv[:,:,:288,:288] * mask[0]
        ensemble_logits = get_logits(X_adv, white_models)
        loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
        loss.backward()
        grad = delta.grad.clone()
        delta.grad.zero_()

        delta.data = delta.data - 1.6/255 * torch.sign(grad)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv



def batch_attack_final_multiple_mask(img,mask,label,white_models):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    mask = mask_final[:,None,:,:]
    mask = torch.cat((mask,mask,mask),1)
    mask = mask.float()
    # print(mask.shape)
    # print(img.shape)
    # mask = F.interpolate(mask, (288, 288), mode='bilinear', align_corners=False)
    mask = mask.cuda()
    
    X_ori = torch.stack([img])
    X = X_ori

    labels = torch.stack([label])
    X.to(device)
    labels.to(device)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    # M_num = int(mask.shape[0]/8)
    cnt = 0
    for t in range(10):
        X_adv = X + delta
        X_adv[:,:,:288,:288] = X_adv[:,:,:288,:288] * mask[cnt]
        cnt += 1
        ensemble_logits = get_logits(X_adv, white_models)
        loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
        loss.backward()
        grad = delta.grad.clone()
        delta.grad.zero_()

        delta.data = delta.data - 1.6/255 * torch.sign(grad)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv


def batch_attack_final_multiple_mask_2(img,mask,label,white_models, M_num = 4, pop_size = 20):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    # mask_final = mask_final[:,:299,:299]
    mask = mask_final[:,None,:,:]
    mask = torch.cat((mask,mask,mask),1)
    mask = mask.float()
    # print(mask.shape)
    # print(img.shape)
    # mask = F.interpolate(mask, (288, 288), mode='bilinear', align_corners=False)
    mask = mask.cuda()
    
    X_ori = torch.stack([img])
    X = X_ori

    labels = torch.stack([label])
    X.to(device)
    labels.to(device)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    # M_num = int(mask.shape[0]/8)
    cnt = 0
    # M_num = 4
    for t in range(10):
        g_temp = []
        for tt in range(M_num):
            # if args.input_diversity:  # use Input Diversity
            X_adv = X + delta
            X_adv[:,:,:288,:288] = X_adv[:,:,:288,:288] * mask[cnt%pop_size]
            cnt += 1

            ensemble_logits = get_logits(X_adv, white_models)
            loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
            loss.backward()
            grad = delta.grad.clone()
            g_temp.append(grad)
        # calculate the mean and cancel out the noise, retained the effective noise
        g = 0.0
        for j in range(M_num):      
            g += g_temp[j]
        delta.grad.zero_()

        delta.data = delta.data - 1.6/255 * torch.sign(g)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv


def batch_attack_final_multiple_mask_2_random(img,mask,label,white_models, M_num = 4, pop_size = 20):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    # mask_final = mask_final[:,:299,:299]
    mask = mask_final[:,None,:,:]
    mask = torch.cat((mask,mask,mask),1)
    mask = mask.float()
    # print(mask.shape)
    # print(img.shape)
    # mask = F.interpolate(mask, (288, 288), mode='bilinear', align_corners=False)
    mask = mask.cuda()
    
    X_ori = torch.stack([img])
    X = X_ori

    labels = torch.stack([label])
    X.to(device)
    labels.to(device)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    # M_num = int(mask.shape[0]/8)
    cnt = 0
    # M_num = 4
    for t in range(10):
        g_temp = []
        for tt in range(M_num):
            # if args.input_diversity:  # use Input Diversity
            X_adv = X + delta
            random_index = np.random.randint(0, pop_size, size=1)
            X_adv[:,:,:288,:288] = X_adv[:,:,:288,:288] * mask[random_index[0]]
            cnt += 1

            ensemble_logits = get_logits(X_adv, white_models)
            loss = -nn.CrossEntropyLoss()(ensemble_logits, labels)
            loss.backward()
            grad = delta.grad.clone()
            g_temp.append(grad)
        # calculate the mean and cancel out the noise, retained the effective noise
        g = 0.0
        for j in range(M_num):      
            g += g_temp[j]
        delta.grad.zero_()

        delta.data = delta.data - 1.6/255 * torch.sign(g)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv


def batch_attack(img,mask,labels,white_models):
    mask_final = torch.zeros([mask.shape[0],mask.shape[1]*patch_size, mask.shape[2]*patch_size],dtype=torch.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]*patch_size):
            for k in range(mask.shape[2]):
                mask_final[i][j][k*patch_size:k*patch_size + patch_size] = mask[i][int(j/patch_size)][int(k)]
    mask = mask_final[:,None,:,:]
    # print(mask.shape)
    mask = torch.cat((mask,mask,mask),1)
    # print(mask.shape)
    # assert False
    mask = mask.float()
    mask = mask.cuda()
    X_ori = torch.squeeze(img)
    # X = torch.stack([mask_X])
    X = X_ori.clone()
    # X = X.type(torch.float32)
    X.to(device)
    labels.to(device)
    # print(labels.shape)
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    grad_momentum = 0
    for t in range(10):
        X_adv = X + delta
        X_adv[:,:,:288,:288] = X_adv[:,:,:288,:288] * mask
        ensemble_logits = get_logits(X_adv, white_models)
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

        delta.data = delta.data - 1.6/255 * torch.sign(grad)
        delta.data = delta.data.clamp(-16/255., 16/255.)
        delta.data = ((X+delta.data).clamp(0.0, 1.0)) - X
    X_adv = X_ori + delta
    return X_adv



def score_transferability(X_adv,label,gray_models):
    labels = label
    # labels = torch.stack([label])
    sum_score = np.zeros((len(gray_models), X_adv.shape[0]))
    model_num = 0
    with torch.no_grad():
        for model in gray_models:
            logits = model(norm(X_adv))
            loss = -nn.CrossEntropyLoss(reduce=False)(logits, labels)
            sum_score[model_num] += loss.detach().cpu().numpy()
            model_num += 1
        Var0 = sum_score.var(axis = 0)
        Mean0 = sum_score.mean(axis = 0)
    # print(Var0)
    # print(Mean0)
    # assert False
    final_sumscore =  Var0 + Mean0
    return final_sumscore



def predict_transfer_score(x,img,label,white_models,gray_models,batch_size=4):
    # 每个个体的得分，通过每个mask单独作用到图像进行对抗攻击产生对抗样本在一组黑盒模型上的效果得分获得
    mask = torch.from_numpy(x)
    mask = mask.reshape(-1,int(HEIGHT/patch_size),int(WIDTH/patch_size))
    # X_adv = single_attack(img,mask,label)  # TODO
    numsum = x.shape[0]
    scorelist = []
    bn = int(np.ceil(numsum/batch_size))
    # print(bn)
    # print(batch_size)
    # print(mask.shape)
    # print(img.shape)
    for i in range(bn):
        bs = i*batch_size
        be = min((i+1)*batch_size, numsum)
        bn = be-bs
        X_adv = batch_attack(torch.stack([img]*bn), mask[bs:be], torch.stack([label]*bn), white_models)
        scorelist = np.append(scorelist,score_transferability(X_adv, torch.stack([label]*bn),gray_models))
    return scorelist




def attack_success():
    return True

class MyDE(GA):
    # 可自定义排序，杂交，变异，选择
    def ranking(self):
        self.Chrom = self.Chrom[np.argsort(self.Y),:]
        self.Y = self.Y[(np.argsort(self.Y))]

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        generation_best_index = self.Y.argmin()
        best_chrom = self.Chrom[generation_best_index]
        best_chrom_Y = self.Y[generation_best_index]
        scale_inbreeding = 0.3
        cross_chrom_size = int(scale_inbreeding * self.size_pop)
        # print(cross_chrom_size)
        superior_size = int(0.3 * self.size_pop)
        generation_superior = self.Chrom[:superior_size,:]
        # half_size_pop = int(size_pop / 2)
        # Chrom1, Chrom2 = self.Chrom[:size_pop,:][:half_size_pop], self.Chrom[:size_pop,:][half_size_pop:]
        self.crossover_Chrom = np.zeros(shape=(cross_chrom_size, len_chrom), dtype=int)
        # print(self.crossover_Chrom.shape)
        for i in range(cross_chrom_size):
            n1 = np.random.randint(0, superior_size, 2)
            # print(n1.shape)
            while n1[0] == n1[1]:
                n1 = np.random.randint(0, superior_size, 2)
            # 让 0 跟多一些
            check_1 = 1
            check_2 = 0
            for j in range(self.len_chrom):
                if generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = 1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = 0
                elif generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_1]][j]
                    check_1 = 1 - check_1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_2]][j]
                    check_2 = 1 - check_2
        return self.crossover_Chrom

 
    def mutation(self):
        scale_inbreeding = 0.3 #+ self.iter/self.max_iter*(0.8-0.2)
        rate = 0.1
        middle_1 = np.zeros((int(self.size_pop*(1-scale_inbreeding)), int(rate * self.len_chrom)))
        middle_2 = np.ones((int(self.size_pop*(1-scale_inbreeding)),self.len_chrom - int(rate * self.len_chrom)))
        self.mutation_Chrom = np.concatenate((middle_1,middle_2), axis=1)
        for i in range(self.mutation_Chrom.shape[0]):
            self.mutation_Chrom[i] = np.random.permutation(self.mutation_Chrom[i])
        return self.mutation_Chrom


    def selection(self, tourn_size=3):
        '''
        greedy selection
        '''
        # 上一代个体Chrom,得分self.Y 
        # 得到这一代个体以及分数
        offspring_Chrom = np.vstack((self.crossover_Chrom,self.mutation_Chrom))
        f_offspring  = self.func(offspring_Chrom)
        # f_chrom = self.Y.copy()
        print("this generate score:")
        print(f_offspring)
        num_inbreeding = int(0.3 * self.size_pop)
        selection_chrom = np.vstack((offspring_Chrom, self.Chrom))
        selection_chrom_Y = np.hstack((f_offspring, self.Y))
        # print(selection_chrom_Y)
        generation_best_index = selection_chrom_Y.argmin()
        # print(selection_chrom[generation_best_index])
        
        
        a, indices = np.unique(selection_chrom_Y, return_index=True)
        # print(a)
        # print(indices)
        
        selection_chrom_1 = np.zeros_like(selection_chrom[0:len(a)])
        selection_chrom_1 = selection_chrom[indices]
        # selection_chrom = selection_chrom[np.argsort(selection_chrom_Y),:]
        # selection_chrom_Y = selection_chrom_Y[(np.argsort(selection_chrom_Y))]
        # print("selection_chrom_1")
        # print(selection_chrom_1)
        if len(a) >= self.size_pop:
            self.Chrom = selection_chrom_1[:self.size_pop,:]
            self.Y = a[:self.size_pop]
        else:
            self.Chrom[0: len(a)] = selection_chrom_1[:len(a),:]
            self.Y[0: len(a)] = a[:len(a)]
            self.Chrom[len(a):self.size_pop] = selection_chrom_1[len(a)-1]
            self.Y[len(a):self.size_pop] = a[len(a)-1]
        # print(self.Chrom[0])
        # assert False



def attack(img, label, white_models, gray_models, maxiter = 10, popsize = 40, width = WIDTH, height = HEIGHT, batch_size = 20):
    # bounds = [(0, 1023)] * (int(width*height/CLUSTER)) + [(0, int(np.math.pow(2, width*height%CLUSTER))-1)]
    bounds = [(0,1)] * int(width/patch_size) * int(height/patch_size)
    # 对于个体中每个元素的范围如[(0, 1)*8] + [(1, 5)*8] 
    # bounds = []
    print('---------begin attack---------------')
    # Format the predict/callback functions for the differential evolution algorithm
    def myfunc(x,img=img,label=label):
        # 评估种群得分的函数
        return predict_transfer_score(x,img,label,white_models,gray_models,batch_size)

    def callback_fn(x, convergence):
        return attack_success()
    
    lb = [0] * len(bounds)
    ub = [elem[1] for elem in bounds]
    # 用的GA模板改成了DE算法
    de = MyDE(func=myfunc, n_dim=len(bounds), size_pop=popsize, max_iter=maxiter, prob_mut=0.001, lb=lb, ub=ub, precision=1, img=None, label=None)


    masks, y = de.run()
    # mask = masks[0]
    # print(masks.shape)
    # assert False
    mask = torch.from_numpy(masks)
    mask = mask.reshape(-1,int(HEIGHT/patch_size),int(WIDTH/patch_size))

    X_adv_esemble_12 = batch_attack_final_multiple_mask_2(img,mask,label,white_models,M_num=12, pop_size=popsize)  # TODO

    X_mask = print_mask(img,mask)


    return X_adv_esemble_12, y, X_mask

