import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def show_pic(img, name, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
#    cv2.imwrite('./1.jpg', img)
#    img = cv2.imread('./1.jpg')
    print(len(bboxes))
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
#    cv2.namedWindow('pic', 0)  # 1表示原图
#    cv2.moveWindow('pic', 0, 0)
#    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
#    cv2.imshow('pic', img)
    cv2.imwrite(name+'.jpg',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 
#    os.remove('./1.jpg')

# 图像均为cv2读取

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
        show_pic(img, file[:-4], coords)    # 原图

        #auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        #show_pic(auged_img, auged_bboxes)  # 强化后的图


