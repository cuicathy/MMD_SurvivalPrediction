# Base / Native
import math
import os
import pickle
import re
import warnings

warnings.filterwarnings('ignore')

from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *

random.seed(10)

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def init_weights(net, init_type='orthogonal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)           # multi-GPUs

    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net


def complete_incomplete_data_selection(opt, data_cv_splits, data_cv_mask, available_idx_file_path, required_modality = ['rad','dna','demo','path']):
    """Modality listed in the required_modality must be available for the selected data, if required_modality is empty [], all data can be used.
    (Data in this task has one modality available at least)
    data_cv_splits, data_cv_mask: data, mask (1 - modality available, 0 - modality missing)
    available_idx_file_path: select the available patient to filter out the data_cv_splits, data_cv_mask
    """
    available_Features_flag = pd.read_csv(available_idx_file_path)
    all_pat_ID = available_Features_flag['TCGA ID'].values

    label_available_list = [1] * len(all_pat_ID) # All data in this dataset has survival time and censer (label) available
    rad_available_list = [1] * len(all_pat_ID)
    dna_available_list = [1] * len(all_pat_ID)
    demo_available_list = [1] * len(all_pat_ID)
    path_available_list = [1] * len(all_pat_ID)

    if ('rad' in required_modality or 'rad_omic' in required_modality):
        rad_available_list = available_Features_flag['rad_available'].values
    if ('dna' in required_modality or 'omic' in required_modality):
        dna_available_list = available_Features_flag['dna_available'].values
    if 'demo' in required_modality:
        demo_available_list = available_Features_flag['demo_available'].values
    if 'path' in required_modality:
        path_available_list = available_Features_flag['path_available'].values
    available_ID_list = [(label_available_list[idx] and rad_available_list[idx] and dna_available_list[idx] and demo_available_list[idx] and path_available_list[idx]) for idx in range(len(rad_available_list))]
    select_available_ID = [all_pat_ID[idx] for idx, i in enumerate(available_ID_list) if (i == 1)]
    print('####number of patients for training and validation', len(select_available_ID))
    ##################################################################################################
    rad_available_list = available_Features_flag['rad_available'].values
    dna_available_list = available_Features_flag['dna_available'].values
    demo_available_list = available_Features_flag['demo_available'].values
    path_available_list = available_Features_flag['path_available'].values
    available_ID_list = [(rad_available_list[idx] and
                          dna_available_list[idx] and demo_available_list[idx] and path_available_list[idx] and label_available_list[idx]) for idx in
                         range(len(rad_available_list))]
    select_available_ID_test = [all_pat_ID[idx] for idx, i in enumerate(available_ID_list) if (i == 1)]
    print('####number of patients for testing', len(select_available_ID_test))

    for k, dataall in data_cv_splits.items():

        print('###### Split %s ######' % k)
        fold_ID = data_cv_splits[k]['train']['x_patname']
        overlap_ID_list = [idx for idx, element in enumerate(fold_ID) if (element in select_available_ID)]

        overlap_ID_name_list = [element for idx, element in enumerate(fold_ID) if
                                (element in select_available_ID)]
        overlap_ID_list_unique_patient = set(overlap_ID_name_list)
        print('#Samples (Training set):',len(overlap_ID_list_unique_patient))
        data_cv_splits[k]['train']['x_omic'] = np.array([dataall['train']['x_omic'][x] for x in overlap_ID_list])
        data_cv_splits[k]['train']['x_rad'] = np.array([data_cv_splits[k]['train']['x_rad'][x] for x in overlap_ID_list])
        data_cv_splits[k]['train']['x_patname'] = [dataall['train']['x_patname'][x] for x in overlap_ID_list]
        data_cv_splits[k]['train']['x_path'] = np.array([dataall['train']['x_path'][x] for x in overlap_ID_list])
        if opt.use_embedding == False:
            data_cv_splits[k]['train']['x_radiomics'] = np.array([data_cv_splits[k]['train']['x_radiomics'][x] for x in overlap_ID_list])
        data_cv_splits[k]['train']['x_demo'] = np.array([dataall['train']['x_demo'][x] for x in overlap_ID_list])
        data_cv_splits[k]['train']['e'] = np.array([dataall['train']['e'][x] for x in overlap_ID_list])
        data_cv_splits[k]['train']['t'] = np.array([dataall['train']['t'][x] for x in overlap_ID_list])

        # Mask
        data_cv_mask[k]['train']['x_path_mask'] = [data_cv_mask[k]['train']['x_path_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['train']['x_omic_mask'] = [data_cv_mask[k]['train']['x_omic_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['train']['x_rad_mask'] = [data_cv_mask[k]['train']['x_rad_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['train']['x_demo_mask'] = [data_cv_mask[k]['train']['x_demo_mask'][x] for x in overlap_ID_list]


        fold_ID = data_cv_splits[k]['val']['x_patname']
        overlap_ID_list = [idx for idx, element in enumerate(fold_ID) if
                           (element in select_available_ID)]
        overlap_ID_name_list = [element for idx, element in enumerate(fold_ID) if
                                (element in select_available_ID)]
        overlap_ID_list_unique_patient = set(overlap_ID_name_list)
        print('#Samples (Validation set):', len(overlap_ID_list_unique_patient))

        data_cv_splits[k]['val']['x_omic'] = np.array([dataall['val']['x_omic'][x] for x in overlap_ID_list])
        data_cv_splits[k]['val']['x_rad'] = np.array([data_cv_splits[k]['val']['x_rad'][x] for x in overlap_ID_list])
        data_cv_splits[k]['val']['x_patname'] = [dataall['val']['x_patname'][x] for x in overlap_ID_list]
        data_cv_splits[k]['val']['x_path'] = np.array([dataall['val']['x_path'][x] for x in overlap_ID_list])
        if opt.use_embedding == False:
            data_cv_splits[k]['val']['x_radiomics'] = np.array([data_cv_splits[k]['val']['x_radiomics'][x] for x in overlap_ID_list])
        data_cv_splits[k]['val']['x_demo'] = np.array([dataall['val']['x_demo'][x] for x in overlap_ID_list])
        data_cv_splits[k]['val']['e'] = np.array([dataall['val']['e'][x] for x in overlap_ID_list])
        data_cv_splits[k]['val']['t'] = np.array([dataall['val']['t'][x] for x in overlap_ID_list])
        # Mask
        data_cv_mask[k]['val']['x_path_mask'] = [data_cv_mask[k]['val']['x_path_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['val']['x_omic_mask'] = [data_cv_mask[k]['val']['x_omic_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['val']['x_rad_mask'] = [data_cv_mask[k]['val']['x_rad_mask'][x] for x in overlap_ID_list]
        data_cv_mask[k]['val']['x_demo_mask'] = [data_cv_mask[k]['val']['x_demo_mask'][x] for x in overlap_ID_list]
    return data_cv_splits, data_cv_mask


def data_aug_mask_multich_4(img, img2, img3, img4):
    seq = iaa.Sequential([
        iaa.LinearContrast(0.9, 1.0),
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-15, 15),
            #shear=(-15, 15),
            order=[0], # nearest interpolation
            mode = ["constant"]
        )
    ])

    # Augment keypoints and images.
    imagelist = []
    imagelist.append(img)
    imagelist.append(img2)
    imagelist.append(img3)
    imagelist.append(img4)

    augDet = seq.to_deterministic()
    row,col = img.shape
    results = np.zeros((4,row,col),float)
    for i in range(4):
        results[i] = augDet.augment_image(imagelist[i])
    return results[0], results[1], results[2], results[3]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    temp = exp_theta * R_mat
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)



def CIndex_lifeline(hazards, labels, survtime_all):
    return (concordance_index(survtime_all, -hazards, labels))


