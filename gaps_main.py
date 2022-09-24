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
from utils.memory import memory_sampling_balanced

import argparser

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from IPython import embed

torch.backends.cudnn.benchmark = True

import torchvision as tv
import torchvision.transforms.functional as tr_F

def crop_partial_img(img_chw, mask_hw, cls_id=1):
    if isinstance(mask_hw, np.ndarray):
        mask_hw = torch.tensor(mask_hw)
    binary_mask_hw = (mask_hw == cls_id)
    binary_mask_hw_np = binary_mask_hw.numpy().astype(np.uint8)
    # RETR_EXTERNAL to keep online the outer contour
    contours, _ = cv2.findContours(binary_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop annotated objects off the image
    # Compute a minimum rectangle containing the object
    assert len(contours) != 0
    cnt = contours[0]
    x_min = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
    x_max = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
    y_min = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
    y_max = tuple(cnt[cnt[:,:,1].argmax()][0])[1]
    for cnt in contours:
        x_min = min(x_min, tuple(cnt[cnt[:,:,0].argmin()][0])[0])
        x_max = max(x_max, tuple(cnt[cnt[:,:,0].argmax()][0])[0])
        y_min = min(y_min, tuple(cnt[cnt[:,:,1].argmin()][0])[1])
        y_max = max(y_max, tuple(cnt[cnt[:,:,1].argmax()][0])[1])
    # Index of max bounding rect are inclusive so need 1 offset
    x_max += 1
    y_max += 1
    # mask_roi is a boolean arrays
    mask_roi = binary_mask_hw[y_min:y_max,x_min:x_max]
    img_roi = img_chw[:,y_min:y_max,x_min:x_max]
    return (img_roi, mask_roi)

def copy_and_paste(novel_img_chw, novel_mask_hw, base_img_chw, base_mask_hw, mask_id):
    base_img_chw = base_img_chw.clone()
    base_mask_hw = base_mask_hw.clone()
    # Horizontal Flipping
    if torch.rand(1) < 0.5:
        novel_img_chw = tr_F.hflip(novel_img_chw)
        novel_mask_hw = tr_F.hflip(novel_mask_hw)
    
    # Parameters for random resizing
    scale = np.random.uniform(0.1, 2.0)
    src_h, src_w = novel_mask_hw.shape
    if src_h * scale > base_mask_hw.shape[0]:
        scale = base_mask_hw.shape[0] / src_h
    if src_w * scale > base_mask_hw.shape[1]:
        scale = base_mask_hw.shape[1] / src_w
    target_H = int(src_h * scale)
    target_W = int(src_w * scale)
    if target_H == 0: target_H = 1
    if target_W == 0: target_W = 1
    # apply
    novel_img_chw = tr_F.resize(novel_img_chw, (target_H, target_W))
    novel_mask_hw = novel_mask_hw.view((1,) + novel_mask_hw.shape)
    novel_mask_hw = tr_F.resize(novel_mask_hw, (target_H, target_W), interpolation=tv.transforms.InterpolationMode.NEAREST)
    novel_mask_hw = novel_mask_hw.view(novel_mask_hw.shape[1:])

    # Random Translation
    h, w = novel_mask_hw.shape
    if base_mask_hw.shape[0] > h and base_mask_hw.shape[1] > w:
        paste_x = torch.randint(low=0, high=base_mask_hw.shape[1] - w, size=(1,))
        paste_y = torch.randint(low=0, high=base_mask_hw.shape[0] - h, size=(1,))
    else:
        paste_x = 0
        paste_y = 0
    
    base_img_chw[:,paste_y:paste_y+h,paste_x:paste_x+w][:,novel_mask_hw] = novel_img_chw[:,novel_mask_hw]
    base_mask_hw[paste_y:paste_y+h,paste_x:paste_x+w][novel_mask_hw] = mask_id

    img_chw = base_img_chw
    mask_hw = base_mask_hw

    return (img_chw, mask_hw)

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
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    dataset_dict['vanilla_train'] = dataset(opts=opts, image_set='train', transform=vanilla_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)
        dataset_dict['vanilla_memory'] = dataset(opts=opts, image_set='memory', transform=vanilla_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs = model(images)
            
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
    return score

class gaps_synthesizer:
    def __init__(self, opts, dataset_dict):
        assert opts.curr_step > 0
        self.old_novel_classes = []
        for novel_step in range(1, opts.curr_step):
            self.old_novel_classes += get_tasks(opts.dataset, opts.task, novel_step)
        self.current_novel_classes = get_tasks(opts.dataset, opts.task, opts.curr_step)
        self.base_classes = get_tasks(opts.dataset, opts.task, 0)
        self.num_base_classes = len(self.base_classes) - 1 # remove bg
        if opts.unknown:
            for i in range(len(self.old_novel_classes)):
                self.old_novel_classes[i] += 1
            for i in range(len(self.current_novel_classes)):
                self.current_novel_classes[i] += 1
        self.partial_data_pool = {}
        self.register_partial_images(dataset_dict)

    def register_partial_images(self, dataset_dict):
        # Register the newest classes
        for cls in self.current_novel_classes:
            self.partial_data_pool[cls] = []
        for i in range(len(dataset_dict['vanilla_train'])):
            # image_chw; mask_hw, sal_map_hw, raw_fn
            img_chw, mask_hw, _, _ = dataset_dict['vanilla_train'][i]
            cur_set = set([i.item() for i in torch.unique(mask_hw)])
            intersection = cur_set.intersection(set(self.current_novel_classes))
            assert len(intersection) > 0
            # train dataset does not have annotations of old classes
            for cls_id_presented in intersection:
                img_roi, mask_roi = crop_partial_img(img_chw, mask_hw, cls_id_presented)
                assert mask_roi.shape[0] > 0 and mask_roi.shape[1] > 0
                self.partial_data_pool[cls_id_presented].append((img_chw, mask_hw, img_roi, mask_roi))
        # Register memory of old classes
        for cls in self.old_novel_classes:
            self.partial_data_pool[cls] = []
        for i in range(len(dataset_dict['vanilla_memory'])):
            img_chw, mask_hw, _, _ = dataset_dict['vanilla_memory'][i]
            if False:
                # Memory class does not consider uncertainty estimate
                mask_hw[mask_hw >= 1] = mask_hw[mask_hw >= 1] + 1
            cur_set = set([i.item() for i in torch.unique(mask_hw)])
            intersection = cur_set.intersection(set(self.old_novel_classes))
            # train dataset does not have annotations of old classes
            for cls_id_presented in intersection:
                img_roi, mask_roi = crop_partial_img(img_chw, mask_hw, cls_id_presented)
                assert mask_roi.shape[0] > 0 and mask_roi.shape[1] > 0
                self.partial_data_pool[cls_id_presented].append((img_chw, mask_hw, img_roi, mask_roi))
    
    def apply_gaps(self, augmented_base_img, augmented_base_label):
        assert augmented_base_img.shape[0] == augmented_base_label.shape[0]
        ret_img_list = []
        ret_mask_list = []
        for b in range(augmented_base_label.shape[0]):
            cur_img_chw = augmented_base_img[b]
            cur_mask_hw = augmented_base_label[b]
            novel_obj_id = random.choice(self.current_novel_classes)
            new_img_chw, new_mask_hw = self.apply_gaps_single(cur_img_chw, cur_mask_hw, novel_obj_id)
            ret_img_list.append(new_img_chw)
            ret_mask_list.append(new_mask_hw)
        return torch.stack(ret_img_list), torch.stack(ret_mask_list)

    def apply_gaps_single(self, syn_img_chw, syn_mask_hw, novel_obj_id):
        # TODO: context-aware sampling is not implemented here
        candidate_classes = [c for c in self.partial_data_pool.keys() if c != novel_obj_id]
        # Gather some useful numbers
        num_novel_classes = len(self.partial_data_pool)
        total_classes = self.num_base_classes + num_novel_classes
        num_novel_instances = len(self.partial_data_pool[novel_obj_id])
        strat = 'vRFS'
        if strat == 'vRFS':
            t = 1. / total_classes
            f_n = num_novel_instances / 500
            r_e = max(1, np.sqrt(t / f_n))
            r_n = r_e
            other_prob = (r_e + r_n) / (r_e + r_n + 1 + r_n)
            selected_novel_prob = (r_n + r_n) / (r_e + r_n + 1 + r_n)
            num_existing_objects = 2
            num_novel_objects = 2
        elif strat == 'always':
            other_prob = 0
            selected_novel_prob = 1
            num_existing_objects = 0
            num_novel_objects = 2
        elif strat == 'CAS':
            other_prob = 1. / total_classes
            selected_novel_prob = 1. / total_classes
            num_existing_objects = 1
            num_novel_objects = 1
        elif strat == 'HALF_HALF':
            other_prob = 0.5
            selected_novel_prob = 0.5
            num_existing_objects = 1
            num_novel_objects = 1
        else:
            raise NotImplementedError("Unknown probabilistic synthesis strategy: {}".format(strat))
        assert other_prob >= 0 and other_prob <= 1
        assert selected_novel_prob >= 0 and selected_novel_prob <= 1

        # Synthesize some other objects other than the selected novel object
        if len(self.partial_data_pool) > 1 and torch.rand(1) < other_prob:
            # select an old class
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_sample = random.choice(self.partial_data_pool[selected_class])
                full_img_chw, full_mask_hw, img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, selected_class)

        # Synthesize selected novel class
        if torch.rand(1) < selected_novel_prob:
            for i in range(num_novel_objects):
                selected_sample = random.choice(self.partial_data_pool[novel_obj_id])
                full_img_chw, full_mask_hw, img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, novel_obj_id)

        return (syn_img_chw, syn_mask_hw)

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
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    if opts.curr_step > 0:
        """ load previous model """
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride, bn_freeze=bn_freeze)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None
    
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)

    print(model.classifier.head)
    
    # Set up optimizer & parameters
    if opts.freeze and opts.curr_step > 0:
        if True:
            for param in model_prev.parameters():
                param.requires_grad = False

            for param in model.parameters():
                param.requires_grad = False
            
            if True:
                for param in model.classifier.parameters():
                    param.requires_grad = True
                
                for param in model.classifier.head[2].parameters():
                    param.requires_grad = False
                
                training_params = [{'params': model.classifier.head[3:-1].parameters(), 'lr': 0.1*opts.lr},
                                    {'params': model.classifier.head[-1].parameters(), 'lr': opts.lr},
                                    {'params': model.classifier.aspp.parameters(), 'lr': 0.01*opts.lr}]

                if opts.curr_step > 1:
                    pass
                    # training_params.append({'params': model.classifier.head[-2].parameters(), 'lr': opts.lr})
                    # for param in model.classifier.head[-2].parameters(): # classifier for new class
                    #     param.requires_grad = True
                
                if opts.unknown:
                    for param in model.classifier.head[0].parameters(): # unknown
                        param.requires_grad = True
                    training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
                    
                    for param in model.classifier.head[1].parameters(): # background
                        param.requires_grad = True
                    training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})
            else:
                for param in model.classifier.head[-1].parameters(): # classifier for new class
                    param.requires_grad = True
                    
                training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': opts.lr}]
                
                if opts.unknown:
                    for param in model.classifier.head[0].parameters(): # unknown
                        param.requires_grad = True
                    training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
                    
                    for param in model.classifier.head[1].parameters(): # background
                        param.requires_grad = True
                    training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})
        
    else:
        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]
        
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=opts.lr, 
                                momentum=0.9, 
                                weight_decay=opts.weight_decay,
                                nesterov=True)
    
    print("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("-----------------------------------------------")
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    if opts.curr_step > 0: # previous step checkpoint
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step-1)
        
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)
        
        if opts.unknown and opts.w_transfer:
            # weight transfer : from unknonw to new-class
            print("... weight transfer")
            curr_head_num = len(model.classifier.head) - 1

            checkpoint[f"classifier.head.{curr_head_num}.0.weight"] = checkpoint["classifier.head.0.0.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.weight"] = checkpoint["classifier.head.0.1.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.bias"] = checkpoint["classifier.head.0.1.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_mean"] = checkpoint["classifier.head.0.1.running_mean"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_var"] = checkpoint["classifier.head.0.1.running_var"]

            last_conv_weight = model.state_dict()[f"classifier.head.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.head.{curr_head_num}.3.bias"]

            for i in range(opts.num_classes[-1]):
                last_conv_weight[i] = checkpoint["classifier.head.0.3.weight"]
                last_conv_bias[i] = checkpoint["classifier.head.0.3.bias"]

            checkpoint[f"classifier.head.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.head.{curr_head_num}.3.bias"] = last_conv_bias

        model.load_state_dict(checkpoint, strict=False)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")

    model = nn.DataParallel(model)
    mode = model.to(device)
    mode.train()
    
    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)
            
        # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    my_gaps_synthesizer = gaps_synthesizer(opts, dataset_dict)
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // 100)
    print(f"... train epoch : {opts.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")
        
    #==========   Train Loop   ==========#
    if opts.test_only:
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
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

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                           reduction='mean')
        
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    
    model.train()
    save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))
    
    # =====  Train  =====
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()
        
        """ data load """
        try:
            images, labels, sal_maps, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, sal_maps, _ = train_iter.next()
            cur_epochs += 1
            avg_loss.reset()
            avg_time.reset()
            
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
        """ memory """
        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()
            
            gaps_rand_idx = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()
            
            m_images[gaps_rand_idx], m_labels[gaps_rand_idx] = my_gaps_synthesizer.apply_gaps(m_images[gaps_rand_idx], m_labels[gaps_rand_idx])

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_sal_maps = m_sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]
            sal_maps[rand_index, ...] = m_sal_maps[rand_index, ...]
        
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=opts.amp):

            outputs = model(images)

            if opts.pseudo and opts.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs_prev = model_prev(images)

                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh), 
                                            pred_labels, 
                                            labels)
                    
                loss = criterion(outputs, pseudo_labels)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        end_time = time.time()

        if (cur_itrs) % 10 == 0:
            print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%6f, Time=%.2f , LR=%.8f" %
                  (opts.task, opts.curr_step, cur_epochs, cur_itrs, total_itrs, 
                   avg_loss.avg, avg_time.avg*1000, optimizer.param_groups[0]['lr']))

        if val_interval > 0 and (cur_itrs) % val_interval == 0:
            print("validation...")
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader, 
                                 device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            
            model.train()
            
            class_iou = list(val_score['Class IoU'].values())
            val_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            print("curr_val_score : %.4f" % (curr_score))
            print()
            
            if curr_score > best_score:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))


    print("... Training Done")
    
    if opts.curr_step > 0:
        print("... Testing Best Model")
        best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(opts.dataset, opts.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':
            
    opts = argparser.get_argparser().parse_args()
        
    start_step = 0 + 1
    total_step = len(get_tasks(opts.dataset, opts.task))
    
    for step in range(start_step, total_step):
        opts.curr_step = step
        main(opts)
        
