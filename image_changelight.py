import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import copy

def _changeLight(img, name):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1) #flag>1为调暗,小于1为调亮  
        exposure.adjust_gamma(img, flag)             
        cv2.imwrite(name+'.jpg',img)
        return exposure.adjust_gamma(img, flag)

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
        _changeLight(img, file[:-4]+'_changelight')    # 原图

        #auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        #show_pic(auged_img, auged_bboxes)  # 强化后的图
