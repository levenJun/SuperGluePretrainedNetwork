#! /usr/bin/env python3
# '''
# Author: leven
# LastEditors: leven
# Description: sp特征提取和sg特征匹配分别独立进行
# '''

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import sys

from demo_superpoint import SuperPointFrontend

from models.superglue import SuperGlue
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

# imgPathPair = ["assets/image_pairs/1403715308062142976.png", "assets/image_pairs/1403715310562142976.png"]
imgPathPair = ["assets/image_pairs/1403715273262142976.png", "assets/image_pairs/1403715310562142976.png"]
# imgPathPair = ["assets/image_pairs/681613500727.pgm", "assets/image_pairs/685679999390.pgm"]
# imgPathPair = ["assets/image_pairs/706045643402.pgm", "assets/image_pairs/706978888714.pgm"]

if __name__ == '__main__':
    # k_thresh = 0.005 #特征提取阈值
    k_thresh = 0.005 #特征提取阈值    
    m_thresh = 0.52   #特征匹配阈值
    device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    # --------------------------------准备工作1：模型初始化------------------------------------
    # 初始化 SuperPoint
    superpoint = SuperPointFrontend(weights_path='models/weights/superpoint_v1.pth',
                                    nms_dist=4,
                                    conf_thresh=k_thresh,
                                    nn_thresh=0.7,
                                    cuda=False)

    # 初始化 SuperGlue
    superglue = SuperGlue({'weights': 'indoor',
                           'match_threshold': m_thresh,  # 设置匹配阈值
                           }).to(device).eval()
    # --------------------------------准备工作1------------------------------------

    # --------------------------------准备工作2：图像读取------------------------------------
    #读取图片数据
    imageBufferPair = [None]*len(imgPathPair)
    img_arrayPair = [None]*len(imgPathPair)
    imageBufferTonsorPair = [None]*len(imgPathPair)
    for idx in range(len(imgPathPair)):
        imageBufferPair[idx] = cv2.imread(imgPathPair[idx], cv2.IMREAD_GRAYSCALE)

        img_arrayPair[idx] = np.array(imageBufferPair[idx])
        print(f"img_array shape = {img_arrayPair[idx].shape}")
        imageBufferTonsorPair[idx] = frame2tensor(img_arrayPair[idx], device)

        cv2.imshow('SuperPoint ori', imageBufferPair[idx] )
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # --------------------------------准备工作2------------------------------------

    # --------------------------------核心步骤1：提取sp特征------------------------------------
    # 提取特征点和描述子
    keypoints0, descriptors0, _ = superpoint.run(np.asarray(imageBufferPair[0]).astype('float32') / 255.)
    keypoints1, descriptors1, _ = superpoint.run(np.asarray(imageBufferPair[1]).astype('float32') / 255.)

    # 假设 keypoints1 和 keypoints2 的形状是 (3, N)
    keypoints0_xy = keypoints0[:2, :].T  # 提取 (x, y) 并转置为 (N, 2)
    keypoints1_xy = keypoints1[:2, :].T

    # 提取置信度分数
    scores0 = keypoints0[2, :]  # 获取第3行，即置信度分数
    scores1 = keypoints1[2, :]

    # 转换为 PyTorch 张量并添加 batch 维度
    keypoints0_xy = torch.from_numpy(keypoints0_xy).float().unsqueeze(0)  # (1, N, 2)
    keypoints1_xy = torch.from_numpy(keypoints1_xy).float().unsqueeze(0)  # (1, N, 2)
    descriptors0 = torch.from_numpy(descriptors0).float().unsqueeze(0)  # (1, 256, N)
    descriptors1 = torch.from_numpy(descriptors1).float().unsqueeze(0)  # (1, 256, N)
    scores0 = torch.from_numpy(scores0).float().unsqueeze(0)  # (1, N)
    scores1 = torch.from_numpy(scores1).float().unsqueeze(0)  # (1, N)
    
    print(f"keypoints0 shape={keypoints0_xy.shape}")
    print(f"descriptors0 shape={descriptors0.shape}")
    print(f"scores0 shape={scores0.shape}")
    # --------------------------------核心步骤1------------------------------------

    # --------------------------------核心步骤2：sg特征匹配------------------------------------
    # 准备数据
    data = {
        'image0': imageBufferTonsorPair[0],
        'image1': imageBufferTonsorPair[1],
        'keypoints0': keypoints0_xy.to(device),
        'keypoints1': keypoints1_xy.to(device),
        'descriptors0': descriptors0.to(device),
        'descriptors1': descriptors1.to(device),
        'scores0': scores0.to(device),
        'scores1': scores1.to(device),
    }

    # 进行匹配
    with torch.no_grad():
        pred = superglue(data)    

    # 匹配完成，获取结果
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    valid = matches > -1

    keypoints0_xy = keypoints0[:2, :].T  # 提取 (x, y) 并转置为 (N, 2)
    keypoints1_xy = keypoints1[:2, :].T

    mkpts0 = keypoints0_xy[valid]
    mkpts1 = keypoints1_xy[matches[valid]]
    # --------------------------------核心步骤2------------------------------------

    # --------------------------------效果显示------------------------------------
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(keypoints0_xy), len(keypoints1_xy)),
        'Matches: {}'.format(len(mkpts0))
    ]

    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(0, 1),
    ]
    out = make_matching_plot_fast(
        imageBufferPair[0], imageBufferPair[1], keypoints0, keypoints1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=False, small_text=small_text)

    cv2.imshow('SuperGlue matches', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # --------------------------------效果显示------------------------------------