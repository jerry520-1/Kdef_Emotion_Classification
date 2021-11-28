from torch.utils.data import Dataset
import os 
import cv2 as cv
import numpy as np
import torch
#----------------------------------------------------#
#在读取图片的时候把图片的预处理加入 原始图片输入562*762 统一转换到762*762
#----------------------------------------------------#
emotion_label = {'AF':0,'AN':1,'DI':2,'HA':3,'NE':4,'SA':5,'SU':6}
def one_hot(x, class_count):
    	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]

def gamma(image):
    image = image/255.0
    gamma = 0.4
    image = np.power(image,gamma)
    return image

def CLHE(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    return image

class Face_Dataset(Dataset):
    def __init__(self,img_dir = None,imgs_transform = None,equalize = False,contrast_enhanced = False):
        self.img_dir   = img_dir
        self.transform = imgs_transform
        self.filelist  = os.listdir(self.img_dir)
        self.equalize  = equalize 
        self.contrast  = contrast_enhanced
        self.a         = 0

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_name   = self.img_dir + self.filelist[index]
        temp_img   = cv.imread(img_name)
        # blank      = np.ones((762,762,3))
        # # x,y,c      = temp_img.shape
        # # if y!=562 or x!=762:
        # temp_img = cv.resize(temp_img,(762,562))

        if self.equalize == True:
            b,g,r    = cv.split(temp_img)
            b1       = cv.equalizeHist(b)
            g1       = cv.equalizeHist(g)
            r1       = cv.equalizeHist(r)
            temp_img = cv.merge([b1,g1,r1])

        if self.contrast == True:
            # temp_img = cv.cvtColor(temp_img,cv.COLOR_RGB2GRAY)
            if self.a == 0:  # 使用伽马变换
                b2,g2,r2    = cv.split(temp_img)
                b2          = gamma(b2)
                g2          = gamma(g2)
                r2          = gamma(r2)
                temp_img = cv.merge([b2,g2,r2])
                self.a   = 1
            else:
                b3,g3,r3    = cv.split(temp_img)
                b3          = CLHE(b3)
                g3          = CLHE(g3)
                r3          = CLHE(r3)
                temp_img = cv.merge([b3,g3,r3])
                self.a   = 0

        label_index= self.filelist[index].split('.')[0][:2]
        emotion    = emotion_label[label_index]
        if self.transform is not None:
            gray_pic = self.transform(temp_img)
        # emotion_one_hot = torch.eye(7)[emotion]
        emotion    = torch.LongTensor([emotion])
        return gray_pic,emotion


class Face_Test_Dataset(Dataset):
    def __init__(self,img_dir = None,imgs_transform = None):
        self.img_dir   = img_dir
        self.transform = imgs_transform
        self.filelist  = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_name   = self.img_dir + self.filelist[index]
        temp_img   = cv.imread(img_name)
        label_index= self.filelist[index].split('.')[0][:2]
        emotion    = emotion_label[label_index]
        if self.transform is not None:
            gray_pic = self.transform(temp_img)
        # emotion_one_hot = torch.eye(7)[emotion]
        emotion    = torch.LongTensor([emotion])
        return gray_pic,emotion
