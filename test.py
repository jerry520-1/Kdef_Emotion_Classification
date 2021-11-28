from operator import index
from unicodedata import normalize
from pandas.core.frame import DataFrame
from torch import nn 
import numpy as np
from torch.autograd.grad_mode import no_grad
from tqdm import tqdm
import torch
import cv2 as cv
import argparse
from torchvision.models import resnet50
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm.cli import main
from datasets import Face_Dataset,Face_Test_Dataset
import pandas as pd
import torchvision as tv


device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512,512))
])
emotion_list = ['afraid','angry','disgust','happy','netural','sad','suprise']
emotion_label = {'AF':0,'AN':1,'DI':2,'HA':3,'NE':4,'SA':5,'SU':6}


'''
  基本功能: 1.测试预测的总体的准确率 
           2.各个类别的混淆矩阵 
           3.输出预测错误的图片id同时保存到指定的文件夹，
           使用 gt 和 pt标注前后差别，并将对应的图片ID输出
'''
def test(args):
    net = nn.Sequential(
        resnet50(),
        nn.Linear(1000,100),
        nn.Linear(100,7)
    )
    model           = torch.nn.DataParallel(net,device_ids= [0,1]).cuda()
    model.load_state_dict(torch.load(args.weights))
    test_datasets   = Face_Test_Dataset(args.test_datasets,img_transforms)
    test_dataloader = DataLoader(test_datasets,batch_size=args.batch_size,num_workers=args.workers,drop_last=False)
    predict_correct_num = 0
    label_num       = 0
    wrong_num       = 0
    target_predict  = []
    target_label    = []
    class_num       = [str(i) for i in range(7)]
    bar             = tqdm(test_dataloader)
    model.eval()
    with no_grad():
         for img ,emotion in tqdm(bar):
            img              = img.to(device)
            emotion_label    = emotion.to(device)
            img              = img.type(torch.FloatTensor)
            # img              = img.unsqueeze(0)
            emotion_predict  = model(img)
            emotion_predict  = emotion_predict.squeeze()
            emotion_predict  = emotion_predict.cpu().numpy()
            emotion_predict  = np.where(emotion_predict == np.max(emotion_predict))
            emotion_label    = emotion_label.cpu().numpy()
            #----------------#
            #计算总体的准确率,并保存预测错误图片和对应id---#
            #----------------#
            label_num       += 1
            if emotion_predict[0] == emotion_label[0]:
                predict_correct_num += 1
            else:
                wrong_num   += 1
                save_wrong_path    = args.predict_wrong + 'origin' + '_' + emotion_list[int(emotion_label[0])] + '_' + 'predict' + '_' + emotion_list[int(emotion_predict[0])] + '__' + str(wrong_num) + '.jpg'
                img          = img.cpu().detach().numpy()
                img          = img.squeeze()
                img          = img.transpose(1,2,0)
                img          = img*255
                img          = img.astype(np.uint8)
                img          = cv.resize(img,(562,762))
                cv.imwrite(save_wrong_path,img)
            #------------------#
            #计算对应的混淆矩阵，并显示出来
            #------------------#
            target_label.append(str(int(emotion_label[0])))
            target_predict.append(str(int(emotion_predict[0])))
            bar.set_description('accuracy_rate of emotion classification is %f' % (predict_correct_num/label_num))
            
    target_label     = np.array(target_label)
    target_predict   = np.array(target_predict)
    matrix     = confusion_matrix(target_label,target_predict)
    dataframe  = pd.DataFrame(matrix,index=emotion_list,columns=emotion_list)
    C_M        = sns.heatmap(dataframe,annot=True,cbar=None,cmap='Blues') 
    plt.title('Confusion Matrix')
    plt.ylabel('emotion_label') 
    plt.xlabel('emotion_predict')
    plt.show()
    C_M = C_M.get_figure()
    C_M.savefig('Confusion_Matrix_Emotion_Classification.jpg',dpi=500)    # 保存混淆矩阵


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',default=16,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--test_datasets',default='./data/KDEF_ORDER_TEST/',type=str)
    parser.add_argument('--weights',default='./weights/EMC100_with_enhanced.pth',type=str)
    parser.add_argument('--predict_wrong',default='./data/KDEF_PREDICT_WRONG/',type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test(args)








