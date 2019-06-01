import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import copy
def _filp_pic_bboxes(img, name, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: #水平翻转
            flip_img =  cv2.flip(flip_img, 1)   #1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])
                cv2.rectangle(flip_img,(int(w-x_max),int(y_min)),(int(w-x_min),int(y_max)),(0,255,0),3)
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min])
                cv2.rectangle(flip_img,(int(x_min),int(h-y_max)),(int(x_max),int(h-y_min)),(0,255,0),3)
        cv2.imwrite(name+'.jpg',flip_img)
        
        return flip_img, flip_bboxes

if __name__ == '__main__':

    ### test ###

    import shutil
    from xml_helper import *

#    need_aug_num = 1                  

#    dataAug = DataAugmentForObjectDetection()

    source_pic_root_path = './data'
    source_xml_root_path = './Annotations'

    
    for _, _, files in os.walk(source_pic_root_path):
        continue
    for file in files:
        pic_path = os.path.join(source_pic_root_path, file)
        xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
        print(xml_path)
        coords = parse_xml(xml_path)        #解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
        coords = [coord[:4] for coord in coords]

        img = cv2.imread(pic_path)
        _filp_pic_bboxes(img, file[:-4]+'_flip',coords)    # 原图

        #auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        #show_pic(auged_img, auged_bboxes)  # 强化后的图
