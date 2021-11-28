import os  
import cv2 as cv
import numpy as np
import random
path = "./data/KDEF/" #文件夹目录
save_path  = './data/KDEF_ORDER/'
# save_path1 = './datasets/KDEF_ORDER1/'
save_path_train  = './data/KDEF_ORDER_TRAIN/'
save_path_test = './data/KDEF_ORDER_TEST/'

files= os.listdir(save_path) #得到文件夹下的所有文件名称  
abr_dict = {'AF':0,'AN':0,'DI':0,'HA':0,'NE':0,'SA':0,'SU':0}
#-------------------------------------------#
#根据文件对应的id读取图片，然后将图片保存到对应的文件
#------------------------------------------#
for file in files: #遍历文件夹  
    for sub_file in os.listdir(path + file):
        temp = sub_file.split('.')
        abr  = temp[0][4:6]
        if abr in abr_dict.keys():
            temp_img  = cv.imread(path + file +'/'+ sub_file)    # 读取对应的图片
            if np.mean(temp_img) > 30:                           # 删除空白图片
                abr_dict[abr] += 1
                file_name = save_path + abr + str(abr_dict[abr]) + '.jpg'
                cv.imwrite(file_name,temp_img)
            else:
                print(path + file + '/' + sub_file)              # 输出图片ID                    
#------------------------------------------#
#划分训练集和测试集
#------------------------------------------#
length = len(files)
rand_seed = random.sample(range(0,length),length)
rand_seed_test = rand_seed[4000:]
rand_seed_train = rand_seed[:4000]
count = 0
for i in rand_seed_train:
    train_read_path = save_path + files[i]
    train_save_path = save_path_train + files[i] 
    img = cv.imread(train_read_path)
    cv.imwrite(train_save_path,img)

for i in rand_seed_test:
    test_read_path = save_path + files[i]
    test_save_path = save_path_test + files[i] 
    img = cv.imread(test_read_path)
    cv.imwrite(test_save_path,img)

# for file in files:
#     count = count + 1
#     read_path = save_path + file
#     read_path











# for file1 in os.listdir(save_path):
#     path_read = save_path + file1
#     path_save = save_path1 + file1
#     img = cv.imread(path_read)
#     if np.mean(img) > 30:
#         cv.imwrite(path_save,img)
#     else:
#         print(path_read)












    #  if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
    #       f = open(path+"/"+file); #打开文件  
    #       iter_f = iter(f); #创建迭代器  
    #       str = ""  
    #       for line in iter_f: #遍历文件，一行行遍历，读取文本  
    #           str = str + line  
    #       s.append(str) #每个文件的文本存到list中  
# print(s) #打印结果  