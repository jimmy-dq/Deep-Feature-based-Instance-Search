"""
Dataset for VID
Written by Heng Fan
"""

from torch.utils.data.dataset import Dataset
import cv2
import os


class Dataset(Dataset):

    def __init__(self, base_path, imgs_list, resnet_transform):
        self.base_path = base_path
        self.imgs_list = imgs_list
        self.transform = resnet_transform

    def __getitem__(self, rand_vid):
        '''
        read a pair of images z and x
        '''
        img_name = self.imgs_list[rand_vid]
        im = cv2.imread(os.path.join(self.base_path, img_name))
        im = self.transform(im)

        return img_name[0:-4], im

    def __len__(self):
        return len(self.imgs_list)
