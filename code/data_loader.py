import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from utils import data_aug_mask_multich_4
import nibabel as nib
from scipy import ndimage, misc
import random

def image2D_resize(image, resize_width, resize_height, interpolation_order):
    width, height = image.shape
    resize_height_scale = resize_height / height
    resize_width_scale = resize_width / width
    image = ndimage.zoom(image, (resize_width_scale, resize_height_scale), order=interpolation_order)
    image = (image - image.min()) / (image.max() - image.min())
    return image


class PathgraphomicDatasetLoader(Dataset):
    def __init__(self, opt, data, mask, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_name = data[split]['x_patname']
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.X_rad = data[split]['x_rad']
        self.X_demo = data[split]['x_demo']
        if opt.use_embedding == False:
            self.x_radiomics = data[split]['x_radiomics']
        self.mask_path = mask[split]['x_path_mask']
        self.mask_omic = mask[split]['x_omic_mask']
        self.mask_demo = mask[split]['x_demo_mask']
        self.mask_rad = mask[split]['x_rad_mask']
        self.opt = opt
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.mode = mode
        self.rad_dir = opt.rad_dir
        self.aug = True
        self.resize_width = 120
        self.resize_height = 120
        self.split = split

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(opt.input_size_path),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transforms_noaug = transforms.Compose([transforms.RandomCrop(opt.input_size_path), transforms.ToTensor(),  # !!!
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transforms_empty= transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)

        single_mask_path = torch.tensor(self.mask_path[index]).type(torch.FloatTensor)
        single_mask_omic = torch.tensor(self.mask_omic[index]).type(torch.LongTensor)
        single_mask_demo = torch.tensor(self.mask_demo[index]).type(torch.FloatTensor)
        single_mask_rad = torch.tensor(self.mask_rad[index]).type(torch.FloatTensor)

        single_X_path = 0
        single_X_omic = 0
        single_X_rad = 0
        single_X_demo = 0
        single_X_radiomics = 0
        single_x_keep_masks = 0

        if self.opt.use_embedding == False:
            if single_mask_path == 1:
                single_X_path = Image.open(self.X_path[index]).convert('RGB')
                if self.split  == 'train':
                    single_X_path = self.transforms(single_X_path)
                else:
                    single_X_path = self.transforms_noaug(single_X_path)
            else:
                single_X_path = self.transforms_empty(Image.new('RGB', (self.opt.input_size_path, self.opt.input_size_path)))
        else:
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)

        single_X_demo = torch.tensor(self.X_demo[index]).type(torch.FloatTensor)
        single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)


        aug_this_image = False
        if (self.aug == True and self.split == 'train'):
            aug_this_image = bool(random.getrandbits(1))

        if self.opt.use_embedding == True: ##0217
            single_X_rad = torch.tensor(self.X_rad[index]).type(torch.FloatTensor)
        else:
            if single_mask_rad != 0 :
                data_name = self.X_rad[index]
                single_X_radiomics = torch.tensor(self.x_radiomics[index]).type(torch.FloatTensor)
                file_path = self.rad_dir + data_name + '_0000.nii.gz'
                image_3d = nib.load(file_path)
                data1 = image_3d.get_data()
                data1 = data1.astype(np.float32)
                data1 = image2D_resize(data1, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0001.nii.gz'
                image_3d = nib.load(file_path)
                data2 = image_3d.get_data()
                data2 = data2.astype(np.float32)
                data2 = image2D_resize(data2, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0002.nii.gz'
                image_3d = nib.load(file_path)
                data3 = image_3d.get_data()
                data3 = data3.astype(np.float32)
                data3 = image2D_resize(data3, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0003.nii.gz'
                image_3d = nib.load(file_path)
                data4 = image_3d.get_data()
                data4 = data4.astype(np.float32)
                data4 = image2D_resize(data4, self.resize_width, self.resize_height, interpolation_order=3)

                if aug_this_image == True:
                    data1, data2, data3, data4 = data_aug_mask_multich_4(data1, data2, data3, data4)
                    data1 = (data1 - data1.min()) / (data1.max() - data1.min())
                    data2 = (data2 - data2.min()) / (data2.max() - data2.min())
                    data3 = (data3 - data3.min()) / (data3.max() - data3.min())
                    data4 = (data4 - data4.min()) / (data4.max() - data4.min())

                data1 = np.expand_dims(data1, axis=0)
                data1_tensor = torch.from_numpy(data1).float()

                data2 = np.expand_dims(data2, axis=0)
                data2_tensor = torch.from_numpy(data2).float()

                data3 = np.expand_dims(data3, axis=0)
                data3_tensor = torch.from_numpy(data3).float()

                data4 = np.expand_dims(data4, axis=0)
                data4_tensor = torch.from_numpy(data4).float()

                multi_tensor = torch.cat((data1_tensor, data2_tensor, data3_tensor, data4_tensor), dim=0)
                single_X_rad = multi_tensor
            else:
                data_name = self.X_rad[index]
                single_X_radiomics = torch.tensor(self.x_radiomics[index]).type(torch.FloatTensor)
                data1 = np.zeros((self.resize_width, self.resize_height))
                data1 = data1.astype(np.float32)

                data2 = np.zeros((self.resize_width, self.resize_height))
                data2 = data2.astype(np.float32)

                data3 = np.zeros((self.resize_width, self.resize_height))
                data3 = data3.astype(np.float32)

                data4 = np.zeros((self.resize_width, self.resize_height))
                data4 = data4.astype(np.float32)

                data1 = np.expand_dims(data1, axis=0)
                data1_tensor = torch.from_numpy(data1).float()

                data2 = np.expand_dims(data2, axis=0)
                data2_tensor = torch.from_numpy(data2).float()

                data3 = np.expand_dims(data3, axis=0)
                data3_tensor = torch.from_numpy(data3).float()

                data4 = np.expand_dims(data4, axis=0)
                data4_tensor = torch.from_numpy(data4).float()

                multi_tensor = torch.cat((data1_tensor, data2_tensor, data3_tensor, data4_tensor), dim=0)
                single_X_rad = multi_tensor


        single_view_masks = {}
        single_view_masks['mask_path'] = single_mask_path
        single_view_masks['mask_rad'] = single_mask_rad
        single_view_masks['mask_demo'] = single_mask_demo
        single_view_masks['mask_omic'] = single_mask_omic

        single_x_keep_masks = torch.stack((single_mask_path, single_mask_rad, single_mask_demo, single_mask_omic)) # Available modality after modal Dropout
        if (self.opt.random_drop_views == True) and (self.split == 'train'):
            nonzero_idx = torch.nonzero(single_x_keep_masks)
            nonzero_idx = list(nonzero_idx)
            if len(nonzero_idx) > 1:
                for i in (nonzero_idx):
                    if random.random() > self.opt.keep_ratio:
                        single_x_keep_masks[i] = 0
                if len(list(torch.nonzero(single_x_keep_masks)))<1:  # If all modalities are dropped out
                    idx = random.sample(nonzero_idx, 1)  # Keep one available modality not be zero
                    single_x_keep_masks[idx] = 1

        single_X_name = self.X_name[index]

        return (single_X_name, single_X_path, single_X_omic, single_X_rad, single_X_demo, single_X_radiomics, single_e, single_t,
                single_view_masks, single_x_keep_masks)
    def __len__(self):
        return len(self.X_path)


class PathgraphomicFastDatasetLoader(Dataset):
    def __init__(self, opt, data, mask, split, mode='omic'):
        self.X_name = data[split]['x_patname']

        self.X_path = data[split]['x_path']
        if opt.use_embedding == False:
            self.X_radiomics = data[split]['x_radiomics']
        self.X_omic = data[split]['x_omic']
        self.X_demo = data[split]['x_demo']
        self.X_rad = data[split]['x_rad']
        self.e = data[split]['e']
        self.t = data[split]['t']

        self.mask_path = mask[split]['x_path_mask']
        self.mask_grph = mask[split]['x_path_mask']
        self.mask_omic = mask[split]['x_omic_mask']
        self.mask_demo = mask[split]['x_demo_mask']
        self.mask_rad = mask[split]['x_rad_mask']

        self.mode = mode
        self.opt = opt
        self.rad_dir = opt.rad_dir
        self.aug = False
        self.split = split
        self.resize_width = 120
        self.resize_height = 120

        self.transforms_noaug = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transforms_empty = transforms.Compose([transforms.ToTensor()])
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transforms_crop = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(opt.input_size_path),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transforms_noaug_crop = transforms.Compose([transforms.RandomCrop(opt.input_size_path), transforms.ToTensor(),  # !!!
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)

        single_mask_path = torch.tensor(self.mask_path[index]).type(torch.FloatTensor)
        single_mask_omic = torch.tensor(self.mask_omic[index]).type(torch.LongTensor)
        single_mask_demo = torch.tensor(self.mask_demo[index]).type(torch.FloatTensor)
        single_mask_rad = torch.tensor(self.mask_rad[index]).type(torch.FloatTensor)

        single_X_path = 0
        single_X_omic = 0
        single_X_rad = 0
        single_X_demo = 0
        single_x_keep_masks = 0
        single_X_radiomics = 0

        if self.opt.use_embedding == False:
            if single_mask_path == 1:
                single_X_path = Image.open(self.X_path[index]).convert('RGB')
                if self.split  == 'train':
                    single_X_path = self.transforms(single_X_path)
                else:
                    single_X_path = self.transforms_noaug(single_X_path)
            else:
                single_X_path = self.transforms_empty(Image.new('RGB', (self.opt.input_size_path, self.opt.input_size_path)))
        else:
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
        single_X_demo = torch.tensor(self.X_demo[index]).type(torch.FloatTensor)
        single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)

        if self.opt.use_embedding == True:
            single_X_rad = torch.tensor(self.X_rad[index]).type(torch.FloatTensor)
        else:
            if single_mask_rad != 0 :
                data_name = self.X_rad[index]
                single_X_radiomics = torch.tensor(self.X_radiomics[index]).type(torch.FloatTensor)  # .squ

                file_path = self.rad_dir + data_name + '_0000.nii.gz'
                image_3d = nib.load(file_path)

                data1 = image_3d.get_data()
                data1 = data1.astype(np.float32)
                data1 = image2D_resize(data1, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0001.nii.gz'
                image_3d = nib.load(file_path)
                data2 = image_3d.get_data()
                data2 = data2.astype(np.float32)
                data2 = image2D_resize(data2, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0002.nii.gz'
                image_3d = nib.load(file_path)
                data3 = image_3d.get_data()
                data3 = data3.astype(np.float32)
                data3 = image2D_resize(data3, self.resize_width, self.resize_height, interpolation_order=3)

                file_path = self.rad_dir + data_name + '_0003.nii.gz'
                image_3d = nib.load(file_path)
                data4 = image_3d.get_data()
                data4 = data4.astype(np.float32)
                data4 = image2D_resize(data4, self.resize_width, self.resize_height, interpolation_order=3)

                data1 = np.expand_dims(data1, axis=0)
                data1_tensor = torch.from_numpy(data1).float()

                data2 = np.expand_dims(data2, axis=0)
                data2_tensor = torch.from_numpy(data2).float()

                data3 = np.expand_dims(data3, axis=0)
                data3_tensor = torch.from_numpy(data3).float()

                data4 = np.expand_dims(data4, axis=0)
                data4_tensor = torch.from_numpy(data4).float()

                multi_tensor = torch.cat((data1_tensor, data2_tensor, data3_tensor, data4_tensor), dim=0)
                single_X_rad = multi_tensor
            else:
                single_X_radiomics = torch.tensor(self.X_radiomics[index]).type(torch.FloatTensor)
                data1 = np.zeros((self.resize_width, self.resize_height))
                data1 = data1.astype(np.float32)

                data2 = np.zeros((self.resize_width, self.resize_height))
                data2 = data2.astype(np.float32)

                data3 = np.zeros((self.resize_width, self.resize_height))
                data3 = data3.astype(np.float32)

                data4 = np.zeros((self.resize_width, self.resize_height))
                data4 = data4.astype(np.float32)

                data1 = np.expand_dims(data1, axis=0)
                data1_tensor = torch.from_numpy(data1).float()

                data2 = np.expand_dims(data2, axis=0)
                data2_tensor = torch.from_numpy(data2).float()

                data3 = np.expand_dims(data3, axis=0)
                data3_tensor = torch.from_numpy(data3).float()

                data4 = np.expand_dims(data4, axis=0)
                data4_tensor = torch.from_numpy(data4).float()

                multi_tensor = torch.cat((data1_tensor, data2_tensor, data3_tensor, data4_tensor), dim=0)
                single_X_rad = multi_tensor

        single_view_masks = {}
        single_view_masks['mask_path'] = single_mask_path
        single_view_masks['mask_rad'] = single_mask_rad
        single_view_masks['mask_demo'] = single_mask_demo
        single_view_masks['mask_omic'] = single_mask_omic

        single_x_keep_masks = torch.stack((single_mask_path, single_mask_rad, single_mask_demo, single_mask_omic))
        if (self.opt.random_drop_views == True) and (self.split == 'train'):
            nonzero_idx = torch.nonzero(single_x_keep_masks)
            nonzero_idx = list(nonzero_idx)
            if len(nonzero_idx) > 1:
                for i in (nonzero_idx):
                    if random.random()>self.opt.keep_ratio: # 50% ratio to drop modalities
                        single_x_keep_masks[i] = 0
                if len(list(torch.nonzero(single_x_keep_masks)))<1: # If all views are 0
                    idx = random.sample(nonzero_idx, 1) # Keep at least one available modality
                    single_x_keep_masks[idx] = 1


        single_X_name = self.X_name[index]

        return (single_X_name, single_X_path, single_X_omic, single_X_rad, single_X_demo, single_X_radiomics, single_e, single_t, single_view_masks, single_x_keep_masks)  # features

    def __len__(self):
        return len(self.X_path)
