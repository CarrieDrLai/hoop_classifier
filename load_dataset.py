# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:35:37 2020

@author: CarrieLai
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np

class load_dataset():
    
    def __init__(self, path_annotation,path_data,crop_size):

        self.crop_size = crop_size
        self.title_list = ['FrameNo', 'x', 'y']

        self.path_annotation = path_annotation
        self.path_data = path_data
        self.tree = ET.parse(self.path_annotation)        
        self.tree = ET.parse(self.path_annotation)
        self.root = self.tree.getroot()

        self.sample_num = [len(list(self.root[i])) for i in range(len(self.root))]
        self.sample_num_all = sum(self.sample_num)
        
        #self.sample_type = []
        self.position = np.zeros([self.sample_num_all,2])
        self.label = np.zeros(self.sample_num_all)
        self.data = [[],[]]


    def load_data(self):
        
        self.img = cv2.imread(self.path_data,0)  

        for sample_type_i in range(len(self.root)):
            for frame_no_i in range(len(self.root[sample_type_i])):
                frame_no = eval(self.root[sample_type_i][frame_no_i].attrib["No"])
                self.label[frame_no] = 1 - sample_type_i
                x = eval(self.root[sample_type_i][frame_no_i][0].text)
                y = eval(self.root[sample_type_i][frame_no_i][1].text)
                self.position[frame_no,0] = x
                self.position[frame_no,1] = y
                
                patch = self.img[y:(y+self.crop_size[1]),x:(x+self.crop_size[0])]
                self.data[1 - sample_type_i].append(patch)                
        for i in range(2):
            self.data[i] = np.reshape(self.data[i],[np.shape(self.data[i])[0],self.crop_size[0],self.crop_size[1]])
        return self.data, self.label
        
#####################################################################
#############################     Main  #############################
#####################################################################

#############################  Parameter  #############################
        
fn_data = 'data.jpg'
fn_annotation = 'annotation.xml'

path_data = '.\\' + fn_data
path_annotation = '.\\' + fn_annotation

crop_size = 96,96

dataset = load_dataset(path_annotation,path_data,crop_size)
data, label = dataset.load_data()
