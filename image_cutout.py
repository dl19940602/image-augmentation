import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import copy

def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

def _cutout(img, name, bboxes, length=500, n_holes=10, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        # 得到h和w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            chongdie = True    #看切割的区域是否与box重叠太多
            
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)    #numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        chongdie = True
                        break
            
            mask[y1: y2, x1: x2, :] = 0.
        
        # mask = np.expand_dims(mask, axis=0)
        img = img * mask
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3)
        cv2.imwrite(name+'.jpg',img)
        return img

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
        _cutout(img, file[:-4]+'_cutout',coords, 500, 10, 0.5)    # 原图

        #auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        #show_pic(auged_img, auged_bboxes)  # 强化后的图
