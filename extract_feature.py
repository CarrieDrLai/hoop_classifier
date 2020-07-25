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

class HoG_descriptor():
    def __init__(self, img, cell_size, bin_size):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img))) # 归一化, gamma=0.5
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient() # 计算每个像素的梯度、角度大小
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), int(self.bin_size)))
        for i in range(cell_gradient_vector.shape[0]): # 为每个cell单元构建梯度方向直方图
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector) # 可视化Cell梯度直方图
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):  # 统计Block的梯度信息
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)

        return hog_vector, hog_image

    def global_gradient(self): # 计算每个像素的梯度、角度大小
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5) # 参数1,0为只在x方向求一阶导数，对x滤波就显示y波形
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5) # 参数0,1为只在y方向求一阶导数，对y滤波就显示x波形
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0) # 实现两幅图片的叠加
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True) # 输出为角度
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient): # 可视化Cell梯度直方图
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image        


class extract_feature:
    
    def __init__(self, dir_save_feature, fn_feature, data,cell_size,bin_size):
        self.dir_save_feature = dir_save_feature
        self.fn_feature = fn_feature
        self.data = data
        self.cell_size = cell_size
        self.bin_size = bin_size
        
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
                hog = HoG_descriptor(temp_img, self.cell_size, self.bin_size)
                vector, image = hog.extract() # vector特征，image图像
                np_vector = np.array(vector)
                final_vector.append(np_vector)
            self.np_final_vector = np.array(final_vector) # (46233, 121, 36)
                # final_image = np.append(final_image, image)
                
            self.make_dir(self.dir_save_feature)
            np.save(self.dir_save_feature + self.fn_feature + ".npy", self.np_final_vector)
    
        return self.np_final_vector