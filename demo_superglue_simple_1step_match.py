#! /usr/bin/env python3
# '''
# Author: leven
# LastEditors: leven
# Description: 基于Matching模块同时完成sp提取和sg匹配
# '''

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import sys

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

# imgPathPair = ["assets/image_pairs/1403715308062142976.png", "assets/image_pairs/1403715310562142976.png"]
imgPathPair = ["assets/image_pairs/1403715273262142976.png", "assets/image_pairs/1403715310562142976.png"]
# imgPathPair = ["assets/image_pairs/681613500727.pgm", "assets/image_pairs/685679999390.pgm"]
# imgPathPair = ["assets/image_pairs/706045643402.pgm", "assets/image_pairs/706978888714.pgm"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.52,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

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

    # --------------------------------准备匹配------------------------------------
    pred = matching({'image0': imageBufferTonsorPair[0], 'image1': imageBufferTonsorPair[1]})
    # --------------------------------完成匹配------------------------------------

    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    print(f"keypoints0 shape={kpts0.shape}, valid.shape={valid.shape}")
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(0, 1),
    ]
    out = make_matching_plot_fast(
        imageBufferPair[0], imageBufferPair[1], kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

    cv2.imshow('SuperGlue matches', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
