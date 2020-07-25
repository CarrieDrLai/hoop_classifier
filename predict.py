# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:45:45 2020

@author: CarrieLai
"""

import numpy as np

class Predict:
    
    def __init__(self, train_feature, test_feature, y_train, y_test, thresh):
        self.train_feature = train_feature
        self.test_feature = test_feature
        self.y_train = y_train
        self.y_test = y_test
        self.thresh = thresh
        

    def sum_featuremap(self,feature_list):
        
        sum_list = []
        for i in range(np.shape(feature_list)[0]):
            sum_list.append(np.sum(feature_list[i]))
        sum_list = np.array(sum_list)
        return sum_list
    
    
    def predict(self):
    
        train_feature_sum = self.sum_featuremap(self.train_feature)
        test_feature_sum = self.sum_featuremap(self.test_feature)
        train_predict = (train_feature_sum - min(train_feature_sum)) / (max(train_feature_sum)-min(train_feature_sum))
        test_predict = (test_feature_sum - min(test_feature_sum)) / (max(test_feature_sum)-min(test_feature_sum))
    
        true1 = sum((train_predict > self.thresh) == self.y_train)
        true2 = sum((train_predict < self.thresh) == self.y_train)
        if true1<true2:
            train_predict = 1 - train_predict
            test_predict = 1 - test_predict
        train_predict = ( train_predict > self.thresh) * 1
        test_predict = ( test_predict > self.thresh) * 1
        
        return train_predict,test_predict