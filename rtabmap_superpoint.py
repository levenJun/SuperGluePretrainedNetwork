#! /usr/bin/env python3
#
# Drop this file in the root folder of SuperPoint git: https://github.com/magicleap/SuperPointPretrainedNetwork
# To use with rtabmap:
#   --Vis/FeatureType 15 --PyDetector/Path "~/SuperPointPretrainedNetwork/rtabmap_superpoint.py" --PyDetector/Model "~/SuperPointPretrainedNetwork/superpoint_v1.pth"
#

import random
import numpy as np
import torch
import cv2
import os

#import sys
#import os
#print(os.sys.path)
#print(sys.version)

from demo_superpoint import SuperPointFrontend

torch.set_grad_enabled(False)

device = 'cpu'
superpoint = []

def init(cuda):
    #print("SuperPoint python init()")
    
    global device
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    
    # 获取当前脚本文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前脚本文件所在的目录
    current_directory = os.path.dirname(current_file_path)    

    # This class runs the SuperPoint network and processes its outputs.
    global superpoint
    superpoint = SuperPointFrontend(weights_path=current_directory+"/models/weights/superpoint_v1.pth",
                          nms_dist=4,
                          conf_thresh=0.015,
                          nn_thresh=1,
                          cuda=cuda)

def detect(imageBuffer):
    #print("SuperPoint python detect()")
    global device
    image = np.asarray(imageBuffer)
    image = (image.astype('float32') / 255.)
   
    global superpoint
    pts, desc, heatmap = superpoint.run(image)
    # return float: Kpts:Nx3, Desc:NxDim
    # use copy to make sure memory is correctly re-ordered
    pts = np.float32(np.transpose(pts)).copy()
    desc = np.float32(np.transpose(desc)).copy()
    print(f"sp detect pts shape={pts.shape}, desc shape={desc.shape}")
    return pts, desc

def detect_test2(imageBuffer, imgRaw):
    print("SuperPoint python detect()")
    global device
    image = np.asarray(imageBuffer)
    image = (image.astype('float32') / 255.)
   
    global superpoint
    pts, desc, heatmap = superpoint.run(image)
    # return float: Kpts:Nx3, Desc:NxDim
    # use copy to make sure memory is correctly re-ordered
    pts = np.float32(np.transpose(pts)).copy()
    desc = np.float32(np.transpose(desc)).copy()
    print(f"sp detect pts shape={pts.shape}, desc shape={desc.shape}")

    # 显示特征点
    for i in range(pts.shape[0]):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        cv2.circle(imgRaw, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('SuperPoint Keypoints', imgRaw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #test
    # init(True)
    init(False) #采用cpu模式
    detect(np.random.rand(640,480)*255) #默认工具函数

    #以下是测试代码
    imgPathList = ["assets/image_pairs/1403715273262142976.png", "assets/image_pairs/1403715310562142976.png"]
    for imgPath in imgPathList:
        # 读取图像
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Image not found or unable to open")

        imgRaw = img.copy()
        img_array = np.array(img)
        detect_test2(img_array, imgRaw)

