# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:35:37 2020

@author: CarrieLai
"""

import cv2
import numpy as np
import math
import os


############################  Hog descriptor  #############################

class HoG(object):
    def __init__(self, img, block_size, block_stride, cell_size, bin_num):
        self.img = img
        self.img_h = img.shape[1]  # h, horizontal
        self.img_v = img.shape[0]  # v, vertical
        self.block_size_h = block_size[1]
        self.block_size_v = block_size[0]
        self.block_stride_h = block_stride[0]
        self.block_stride_v = block_stride[1]
        self.cell_size_h = cell_size[1]
        self.cell_size_v = cell_size[0]
        self.bin_num = bin_num
        self.bin_unit = 360 / bin_num

    def gradient(self):
        self.img = np.sqrt(self.img / np.max(self.img))  # gamma normalization

        dx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
        self.magnitude = abs(magnitude)
        self.phase = cv2.phase(dx, dy, angleInDegrees=True)

    # calculate bins for every cell
    def bins(self, cell_mag, cell_phase):
        bins = np.zeros(self.bin_num)
        for i in range(self.cell_size_v):
            for j in range(self.cell_size_h):
                quotient = int(cell_phase[i][j] / self.bin_unit)
                mod = cell_phase[i][j] % self.bin_unit
                if quotient == self.bin_num:
                    quotient = self.bin_num-1                
                bins[quotient] += cell_mag[i][j] * (1 - mod / self.bin_unit)
                if quotient == self.bin_num - 1:
                    bins[0] += cell_mag[i][j] * (mod / self.bin_unit)
                else:
                    bins[quotient + 1] += cell_mag[i][j] * (mod / self.bin_unit)
        return bins

    # get cell vectors(3D)
    def cell_feature(self):
        self.cell_vector = np.zeros([int(self.img_v / self.cell_size_v), int(self.img_h / self.cell_size_h),
                                self.bin_num])
        for i in range(self.cell_vector.shape[0]):
            for j in range(self.cell_vector.shape[1]):
                cell_mag = self.magnitude[self.cell_size_v * i: self.cell_size_v * (i + 1),
                           self.cell_size_h * j: self.cell_size_h * (j + 1)]
                cell_phase = self.phase[self.cell_size_v * i: self.cell_size_v * (i + 1),
                             self.cell_size_h * j: self.cell_size_h * (j + 1)]
                self.cell_vector[i][j] = self.bins(cell_mag, cell_phase)

    # normalize block vector
    def norm_block(self, vector):
        sum = 0
        for i in vector:
            sum += i ** 2
        num = math.sqrt(sum)
        if num != 0:
            vector /= num
        return vector

    # get hog vector
    def hog_feature(self):
        self.cell_feature()
        hog_vector = []
        block_num_h = int((self.img_h - self.block_size_h) / self.block_stride_h + 1)
        block_num_v = int((self.img_v - self.block_size_v) / self.block_stride_v + 1)
        bczRate_h = int(self.block_size_h / self.cell_size_h)  # block cell size rate
        bczRate_v = int(self.block_size_v / self.cell_size_v)
        bcsRate_h = int(self.block_stride_h / self.cell_size_h)  # block cell stride rate
        bcsRate_v = int(self.block_stride_v / self.cell_size_v)
        for i in range(block_num_v):
            for j in range(block_num_h):
                block_vector = self.cell_vector[bcsRate_v * i : bcsRate_v * i + bczRate_v,
                               bcsRate_h * j : bcsRate_h * j + bczRate_h]
                block_vector = block_vector.reshape(bczRate_h * bczRate_v * self.bin_num, 1)
                block_vector = self.norm_block(block_vector)
                hog_vector.extend(block_vector)
        return hog_vector

    # get image
    def hog_image(self):
        hog_img = np.zeros([self.img_v, self.img_h])
        max = np.array(self.cell_vector).max()
        halfcell_h = self.cell_size_h / 2
        halfcell_v = self.cell_size_v / 2
        for i in range(int(self.img_v / self.cell_size_v)):
            for j in range(int(self.img_h / self.cell_size_h)):
                cell_grad = self.cell_vector[i][j] / max
                angle = 0
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(i * self.cell_size_h + halfcell_h + magnitude * halfcell_h * math.cos(angle_radian))
                    y1 = int(j * self.cell_size_v + halfcell_v + magnitude * halfcell_v * math.sin(angle_radian))
                    x2 = int(i * self.cell_size_h + halfcell_h - magnitude * halfcell_h * math.cos(angle_radian))
                    y2 = int(j * self.cell_size_v + halfcell_v - magnitude * halfcell_v * math.sin(angle_radian))
                    cv2.line(hog_img, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += self.bin_unit
        return hog_img

    def hog_extract(self):
        self.gradient()
        hog_vector = self.hog_feature()
        hog_img = self.hog_image()
        return hog_vector, hog_img      


class extract_feature:
    
    def __init__(self, dir_save_feature, fn_feature, data, block_size, block_stride, cell_size, bin_num):
        self.dir_save_feature = dir_save_feature
        self.fn_feature = fn_feature
        self.data = data
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.bin_num = bin_num
        
    def make_dir(self,path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    
    def HoG_output_vector(self):
        
        if os.path.exists(self.dir_save_feature + self.fn_feature + ".npy"):
            self.np_final_vector = np.load(self.dir_save_feature + self.fn_feature + ".npy")
            print(" =========  " + self.fn_feature + "Feature File is already exist  =========")
        else:            
            final_vector = []
            #final_image = []
            
            for frame in range(np.shape(self.data)[0]): # for frame in range(label.shape[0]):
                if frame % 3000 == 0:
                    print(" >>>> Extracte Frame No." + str(frame))
                temp_img = self.data[frame, :, :]
                hog = HoG(temp_img, self.block_size, self.block_stride, self.cell_size, self.bin_num)
                vector, image = hog.hog_extract() # vector特征，image图像
                np_vector = np.array(vector)
                final_vector.append(np_vector)
            self.np_final_vector = np.array(final_vector) # (46233, 121, 36)
                # final_image = np.append(final_image, image)
                
            self.make_dir(self.dir_save_feature)
            np.save(self.dir_save_feature + self.fn_feature + ".npy", self.np_final_vector)
    
        return self.np_final_vector