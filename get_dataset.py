# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:12:23 2020

@author: CarrieLai
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

############################### Get crop center ############################

class GetHoopPos:
    def __init__(self,fn_video,crop_size):
        self.video_dir = '.\\' + fn_video
        self.crop_size = crop_size
        self.cap = cv2.VideoCapture(self.video_dir) 
        self.success,self.frame = self.cap.read()
        #self.get_pos()
        print(" ============== GetHoopPos is done ==============\n")

    def draw_rectangle(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            x1, y1 = x - int(self.crop_size[0]/2), y - int(self.crop_size[1]/2)
            x2, y2 = x + int(self.crop_size[0]/2), y + int(self.crop_size[1]/2)
            cv2.rectangle(self.frame,(x1,y1),(x2,y2),(0, 255, 0), 8)  
            
            self.HoopPos.append((x,y))
            print(" ===========   The CenterPos of Hoop ============\n" + "       =======   x : " + str(x) + " y :" + str(y) + "  =======")
        
    
    def get_pos(self):
        self.HoopPos = []
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.draw_rectangle,self.HoopPos)

        while (1):
            cv2.imshow('frame', self.frame)
            if cv2.waitKey(40) & 0xFF == 27:   ### 25Hz : 40ms; 0xff：取按键后8位； 27：Esc退出
                break
        cv2.destroyWindow("frame")
        self.cap.release()
        return self.HoopPos
#    def return_pos(self):
#        return self.HoopPos
    

############################### Get Data ############################
        
class GetDataset:
    def __init__(self,video_dir,save_data_path,dir_pos, dir_neg, crop_size, hoop_pos,colume):
        self.video_dir = video_dir
        self.save_data_npy = '.\\data.npy'
        self.save_data_path = save_data_path
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        self.make_dir(self.dir_pos)
        self.make_dir(self.dir_neg)
        
        self.crop_size = crop_size
        self.hoop_pos = hoop_pos
        self.colume = colume
        self.separate_frame_count = 5000
        
        self.cap = cv2.VideoCapture(self.video_dir) 
        self.dataset = []

    def make_dir(self,path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

    def save_data(self):
        if os.path.exists(self.save_data_path):
            print(" =========  Data Img is already exist  =========")
        else:
            self.width = self.colume * self.crop_size[0]
            self.height = int(np.ceil(self.count/self.colume)) * self.crop_size[1]
            self.result = np.zeros([self.height, self.width],np.uint8)
            
            for i, im in enumerate(self.dataset):
                w = int(np.mod(i,self.colume) * self.crop_size[0])
                h = int(i/self.colume) * self.crop_size[1]
                self.result[h:(h + self.crop_size[1]),w:(w + self.crop_size[0])] = im
                if np.mod(i,self.colume*30)==0:    
                    print(" >>>>>> Saving data No." + str(i))
            cv2.imwrite(self.save_data_path,self.result)
            print(" >>>>>> Saving Data !!!!!!  Suceess  !!!!!! ") 

    
    def get_data(self):
        
        self.count = 0
        if os.path.exists(self.save_data_npy):
            self.dataset = np.load(self.save_data_npy)
            self.count = np.shape(self.dataset)[0]
            print(" =========  Data File is already exist  =========")
        else:
            while self.cap.isOpened():
                self.success,self.frame = self.cap.read() 
    
                if self.success is False :
                    self.cap.release()
                    break
                if (self.count % (self.colume*30) == 0):
                    print(">>>>>> Geting data No." + str(self.count))
                
                self.count += 1
                x1, y1 = self.hoop_pos[0][0] - int(self.crop_size[0]/2), self.hoop_pos[0][1] - int(self.crop_size[1]/2)
                x2, y2 = self.hoop_pos[0][0] + int(self.crop_size[0]/2), self.hoop_pos[0][1] + int(self.crop_size[1]/2)
                self.dataset.append(cv2.cvtColor(self.frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY))
                cv2.waitKey(1) 
            self.dataset =  np.reshape(self.dataset,[self.count,self.crop_size[0],self.crop_size[1]])
            np.save(self.save_data_npy, self.dataset) 
            print(" >>>>>> Getting data !!!!!!  Suceess  !!!!!! \n")
        
        self.save_data()
        
        return self.dataset        
            
    ############################### Seperate Data Roughly ############################

    def rough_separate(self):
        
        self.gray = np.sum(np.reshape(self.dataset,[np.shape(self.dataset)[0],self.crop_size[0]*self.crop_size[1]]),1)/(self.crop_size[0]*self.crop_size[1])
    
        thresh = (np.min(self.gray[0:5000])+np.max(self.gray[0:5000]))/2
        mask = self.gray<thresh
        if np.size(os.listdir(self.dir_pos))>0:
            print("=======  Rough Seperation is already done. =======")
            return
        for i in range(np.shape(self.dataset)[0]):

            if i>0 and (i%self.separate_frame_count)==0:
                thresh = (np.min(self.gray[(i-self.separate_frame_count):i])+np.max(self.gray[(i-self.separate_frame_count):i]))/2
                mask = self.gray < thresh
            if mask[i]==True:
                cv2.imwrite(self.dir_pos + str(i)+".jpg", self.dataset[i])
            else:
                cv2.imwrite(self.dir_neg + str(i)+".jpg", self.dataset[i])
        print(" >>>>>> Rough Seperation !!!!!!  Suceess  !!!!!! ") 
                
                
############################### Make Annotation File ############################

class create_annotation():
    
    def __init__(self,dir_pos, dir_neg, fn_annotation, dataset, crop_size, colume):
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        self.save_annotation_dir = '.\\' + fn_annotation
        self.save_label_npy = '.\\label.npy'
        
        self.sample_type = ['Positive','Negetive']
        self.title_list = ['FrameNo', 'x', 'y']
        
        self.crop_size = crop_size
        self.colume = colume
        
        self.dataset = dataset
        self.list_pos = self.get_sample_list(self.dir_pos)
        self.get_label_list()
        self.label_neg = 1 - self.label
        self.pos_num = sum(self.label)
        
        self.creater_annotation_file()
        
    def get_sample_list(self,sample_dir):
        list_sample = [os.listdir(sample_dir)[i].split(".")[0] for i in range(len(os.listdir(sample_dir)))]
        list_sample = np.uint32(list_sample)
        list_sample = np.msort(list_sample)
        return list_sample
    
    def get_label_list(self):
        if os.path.exists(self.save_label_npy):
            self.label = np.load(self.save_label_npy)
            print(" =========  Label File is already exist  =========")    
        else:
            self.label = np.zeros([np.shape(self.dataset)[0]])
            for i in range(len(self.list_pos)):
                self.label[self.list_pos[i]]=1
            self.label = np.uint8(self.label)
            np.save(self.save_label_npy, self.label)
            print(" >>>>>> Save Label File !!!!!!  Suceess  !!!!!! ") 
    
    def creater_annotation_file(self):
        
        if os.path.exists(self.save_annotation_dir):
            print(" =========  Annotataion File is already exist  =========")
        else:
            print(" >>>>>> Create Annotation File !!!!!!  Start  !!!!!! ") 
            root = ET.Element('Annotation')
            root.text = '\n' + 1 * '\t'
    
            nodes = []
            for i in range(len(self.sample_type)):
                node_type = ET.SubElement(root, self.sample_type[i])
                node_type.text = '\n' + 2 * '\t'
                nodes.append(node_type)
                if i == len(self.sample_type)-1:
                    nodes[i].tail = '\n' 
                else:
                    nodes[i].tail = '\n' + 1 * '\t'
    
            for node_i in range(len(self.label)):
                if (node_i % 5000) == 0:
                    print(" >>>>>> Saving Frame No." + str(node_i))
                    
                label_i = self.label[node_i]
                frame_id = node_i
                frame = ET.SubElement(nodes[1-label_i], self.title_list[0])
                frame.set('No',str(node_i))
    
                if sum(self.label[:(node_i+1)]) == self.pos_num or sum(self.label_neg[:(node_i+1)]) == (len(self.label)-self.pos_num):
                    frame.tail = '\n' + 1 * '\t'
                else:
                    frame.tail = '\n' + 2 * '\t'
        
                x = np.mod(frame_id, self.colume) * self.crop_size[0]
                y = int(frame_id/self.colume) * self.crop_size[1]
                p = [] 
                p.append(x)
                p.append(y)
                for title_id in range(1,3):
                    currTitle = self.title_list[title_id]
                    position = ET.SubElement(frame, currTitle)
                    position.text = str(p[title_id-1])
    
            tree = ET.ElementTree(root)
            tree.write(self.save_annotation_dir, encoding="utf-8", xml_declaration=True)
            print(" >>>>>> Create Annotation File !!!!!!  Suceess  !!!!!! ") 

    
    def return_(self):
        return self.label

