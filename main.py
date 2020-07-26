# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:44:23 2020

@author: CarrieLai
"""

from get_dataset import GetHoopPos,GetDataset,create_annotation
from load_dataset import load_dataset
from extract_feature import extract_feature
from predict import Predict

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#############################   Read Congig File   #############################

print(" ============== Read Config File ==============\n")
f = open('config.txt','r')
config = f.readlines()
f.close()

var_list=[]
for var in config:
    var = var.strip()
    var_name = var.split("=")[0].strip()
    var_value = var.split("=")[1].strip()
    if var_name == 'crop_size' or var_name == 'colume' or var_name == 'block_size' or var_name == 'block_stride' or var_name == 'cell_size' or var_name == 'bin_num' or var_name == 'thresh':
        var_value = eval(var_value)
    var_list.append([var_name, var_value])
    print( " " + var_name + " = " + str(var_value))
var_list = dict(var_list)

print(" ============== Read Config File Success ==============\n")

#############################        Parameter        ###################################

fn_video = var_list["fn_video"]
fn_data = var_list["fn_data"]
fn_annotation = var_list["fn_annotation"]
dir_pos = var_list["dir_pos"]
dir_neg = var_list["dir_neg"]
dir_save_feature = var_list["dir_save_feature"]
crop_size = var_list["crop_size"]
colume = var_list["colume"]

block_size = var_list["block_size"]
block_stride = var_list["block_stride"]
cell_size = var_list["cell_size"]
bin_num = var_list["bin_num"]

thresh = var_list["thresh"]
task_type = var_list["task_type"]
# 1. locate: get HoopPos
# 2. crop : get all patch, save data.jpg and rough separete(and then separate manually)  
# 3. annotation
# 4. train 
# 5. result

dir_video = ".\\" + fn_video
path_data = ".\\" + fn_data
path_annotation = ".\\" + fn_annotation

#############################  Step 1 : Prepare Dataset  ################################

if task_type == 'locate': 

    ################  (a)Get Hoop Position
    hoop = GetHoopPos(fn_video,crop_size)
    hoop_pos = hoop.get_pos()        #hoop_pos = [(924,133)]

elif task_type == 'crop':

    ################   (b)Get all data & save data as npy
    Data = GetDataset(fn_video, path_data,dir_pos, dir_neg, crop_size,hoop_pos,colume)  
    patch_all = Data.get_data()
    Data.rough_separate()
    ################  (c)Manually seperate the data after rough seperation 
    
elif task_type == 'annotation':
    ################  (d)Make Annotation File
    create_annotation(dir_pos, dir_neg, fn_annotation, patch_all, crop_size, colume)

###########################    Step 2 : Train Model   ##############################

elif task_type == 'train':
    
    ###############  (a)load dataset

    Dataset = load_dataset(path_annotation,path_data,crop_size)
    data, label = Dataset.load_data() 
    X_train,X_test, y_train, y_test = train_test_split(data,label,test_size=0.3, random_state=0)

    ###############  (b)Extract Feature  

    hog_train = extract_feature(dir_save_feature, "X_train", X_train, block_size, block_stride, cell_size, bin_num)
    train_feature = hog_train.HoG_output_vector()
    print("\n >>>>>> Extract Train Feature !!!!!!  Suceess  !!!!!! \n") 
    hog_test = extract_feature(dir_save_feature, "X_test", X_test[0:1], block_size, block_stride, cell_size, bin_num)
    test_feature = hog_test.HoG_output_vector()
    print("\n >>>>>> Extract Test Feature !!!!!!  Suceess  !!!!!! \n") 

elif task_type == 'result':

    ##############  (c)Fit Model & Predict  
    TPR = []
    FPR = []
    for t in range(10,37):
        thresh = t/50
        Pre = Predict(train_feature, test_feature, y_train, y_test, thresh)
        train_predict, test_predict = Pre.predict()

###########################    Step 3 : Analysis    ##############################
    
        #ROC曲线
        true_pos = sum((y_test == 1) * test_predict)
        true_neg = sum((y_test == 0) * (test_predict == 0))
        false_pos = sum((y_test == 0) * (test_predict == 1))
        false_neg = sum((y_test == 1) * (test_predict == 0))
        TPR.append(true_pos/(false_neg + true_pos))
        FPR.append(false_pos/(false_pos + true_neg))# False Positive Rate

    plt.figure()
    plt.plot(FPR,TPR,'r--',5)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.show()