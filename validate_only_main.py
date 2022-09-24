"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from IPython import embed

torch.backends.cudnn.benchmark = True

import torchvision as tv
import torchvision.transforms.functional as tr_F
import argparser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    vanilla_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)

    return dataset_dict

def jit_validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs = model(images)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
    return score

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
        
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: # re-labeling: [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)
            
    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Dataset: %s, Test set: %d" % (opts.dataset, len(dataset_dict['test'])))
    
    #==========   TEST ONLY   ==========#
    # change jit trace weights here
    my_model = torch.jit.load("/tmp/base_15.pt")
    test_score = jit_validate(opts=opts, model=my_model, loader=test_loader, 
                            device=device, metrics=metrics)
    
    print(metrics.to_str(test_score))
    class_iou = list(test_score['Class IoU'].values())
    class_acc = list(test_score['Class Acc'].values())

    first_cls = len(get_tasks(opts.dataset, opts.task, 0)) # 15-1 task -> first_cls=16
    print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
    print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
    return


if __name__ == '__main__':
            
    opts = argparser.get_argparser().parse_args()
        
    opts.curr_step = 5 # STEP to test
    main(opts)
        
