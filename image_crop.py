import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import copy

def _crop_img_bboxes(img, name, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- 裁剪boundingbox ----------------------
        #裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])
            cv2.rectangle(crop_img,(int(bbox[0]-crop_x_min),int(bbox[1]-crop_y_min)),(int(bbox[2]-crop_x_min),int(bbox[3]-crop_y_min)),(0,255,0),3)
        cv2.imwrite(name+'.jpg',crop_img)
        
        return crop_img, crop_bboxes
  
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
        _crop_img_bboxes(img, file[:-4]+'_crop',coords)    # 原图

        #auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        #show_pic(auged_img, auged_bboxes)  # 强化后的图
