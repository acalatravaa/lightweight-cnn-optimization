"""
Executes the training and validation loops for the MobileNetV2-YOLO architecture 
on the PASCAL VOC dataset, logging Mean Average Precision (mAP) metrics.
"""

import argparse
import os
import random
import shutil
import time
import yaml
from pprint import PrettyPrinter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from progress.bar import IncrementalBar

import folder2lmdb
from models.voc.mbv2_yolo import yolo
from models.voc.yolo_loss import *
from utils import Logger, AverageMeter
from utils.eval_mAP import *

pp = PrettyPrinter()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch YOLO Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='Initial learning rate') 
    parser.add_argument('--warm-up', default=[], type=float, nargs='*', help='Warm up epochs')                    
    parser.add_argument('--epochs', default=20, type=int, help='Total epochs to run')
    parser.add_argument('--schedule', type=int, nargs='+', default=[6, 10, 14], help='Epochs to decrease LR')
    parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, help='Directory to save checkpoints')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate model on validation set only') 
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(10992)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing training on device: {device}")

    # Load Configurations
    with open('models/voc/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) 

    with open('data/voc_data.yaml', 'r') as f:
        dataset_path = yaml.load(f, Loader=yaml.Loader)     

    # Setup Data Loaders
    image_folder = folder2lmdb.ImageFolderLMDB                                 
    train_dataset = image_folder(
        db_path=dataset_path["trainval_dataset_path"]["lmdb"],
        transform_size=config["train_img_size"],
        phase='train'
    )       
    test_dataset = image_folder(
        db_path=dataset_path["test_dataset_path"]["lmdb"],
        transform_size=[[config["img_w"], config["img_h"]]],
        phase='test'
    )    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, config["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn
    ) 
    
    # Initialize Model and Optimizer
    model = yolo(config=config).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)   

    os.makedirs(args.checkpoint, exist_ok=True)  

    start_epoch = 0
    best_acc = 0.0

    # Resume Logic
    if args.resume:
        print(f"[*] Resuming from checkpoint: {args.resume}")
        if not os.path.isfile(args.resume):
            raise FileNotFoundError("Checkpoint not found!")
            
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.yolo_losses[0].val_conf = checkpoint['conf'] 
        model.yolo_losses[1].val_conf = checkpoint['conf'] 
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='voc-training', resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='voc-training')
        logger.set_names(['Epoch', 'Loss', 'Precision', 'Time', 'IOU', 'Learning Rate'])

    # Evaluation Only Mode
    if args.evaluate:
        test(test_loader, model, device, config)
        return
        
    # Main Training Loop
    test_acc = 0 
    for epoch in range(start_epoch, args.epochs):
        
        if epoch in args.warm_up: 
            adjust_learning_rate(optimizer, 10)
            
        if epoch in args.schedule:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'conf': model.yolo_losses[0].val_conf,
            }, False, args.checkpoint, filename=f'epoch{epoch}_checkpoint.pth.tar') 
            
            adjust_learning_rate(optimizer, 0.5)
            
        log_epoch = (epoch % 2 == 0)
        if log_epoch:
            print(f"\nEpoch: [{epoch:3d} | {args.epochs:3d}] LR: {optimizer.param_groups[0]['lr']:f}")
            start_time = time.time()
        
        train_loss, iou = train_epoch(train_loader, model, optimizer, device)
        
        if not log_epoch:
            test_acc = test(test_loader, model, device, config)  
            logger.append([epoch + 1, train_loss, test_acc, time.time() - start_time, iou, optimizer.param_groups[0]['lr']])
            
            # Save best model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc) 
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'conf': model.yolo_losses[0].val_conf,
            }, is_best, args.checkpoint)


def train_epoch(train_loader, model, optimizer, device):
    """Executes a single epoch of training."""
    model.train()
    bar = IncrementalBar('Training', max=len(train_loader), width=12)
    losses = AverageMeter()
    iou = [AverageMeter(), AverageMeter()]

    for batch_idx, (images, targets) in enumerate(train_loader):
        bs = images.size(0)
        images = images.to(device) 
        
        optimizer.zero_grad()
        outputs = model(images, targets)
        
        t_loss = list()
        for i, l in enumerate(outputs):
            t_loss.append(l[0])  
            iou[i].update(l[2])

        loss = sum(t_loss)
        losses.update(loss.item(), bs)
        loss.backward()
        optimizer.step()

        bar.suffix = f'{bar.elapsed_td} | Loss: {losses.avg:.4f} | IOU1: {iou[0].avg:.3f} | IOU2: {iou[1].avg:.3f}'
        bar.next()
        
    bar.finish()
    return losses.avg, (iou[0].avg + iou[1].avg) / 2
    

def test(test_loader, model, device, config):
    """Evaluates the model on the validation dataset and computes mAP."""
    model.eval()
    n_classes = config['yolo']['classes']
    
    bar = IncrementalBar('Validating', max=len(test_loader), width=32)
    
    det_boxes, det_labels, det_scores = [], [], []
    true_boxes, true_labels, true_difficulties = [], [], []
    gt_box_count, pred_box_count = 0, 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)     
            labels = [torch.Tensor(l).to(device) for l in targets] 
            bs = len(labels)
            
            detections = model(images)  
            
            for sample_i in range(bs):
                target_sample = labels[sample_i]
                gt_box_count += len(target_sample)
                
                # Ground truth processing
                tx1, tx2 = torch.unsqueeze((target_sample[...,1] - target_sample[...,3] / 2),1), torch.unsqueeze((target_sample[...,1] + target_sample[...,3] / 2),1)
                ty1, ty2 = torch.unsqueeze((target_sample[...,2] - target_sample[...,4] / 2),1), torch.unsqueeze((target_sample[...,2] + target_sample[...,4] / 2),1)
                box = torch.cat((tx1, ty1, tx2, ty2), 1)
 
                true_boxes.append(box)
                true_labels.append(target_sample[..., 0])
                true_difficulties.append(torch.zeros(target_sample.size(0), requires_grad=False))

                # Prediction processing
                preds = detections[sample_i]
                pred_box_count += len(preds) if preds is not None else 0
                
                if preds is not None:                                
                    det_boxes.append(preds[...,:4])
                    det_labels.append((preds[...,6] + 1).to(device))
                    det_scores.append((preds[...,4] * preds[...,5]).to(device))
                else:
                    empty = torch.empty(0).to(device)
                    det_boxes.append(empty)
                    det_labels.append(empty)
                    det_scores.append(empty)
                    
            bar.next()
            
    bar.finish()
    
    # Adjust dynamic confidence thresholds
    model.yolo_losses[0].val_conf = adjust_confidence(gt_box_count, pred_box_count, model.yolo_losses[0].val_conf)
    model.yolo_losses[1].val_conf = adjust_confidence(gt_box_count, pred_box_count, model.yolo_losses[1].val_conf)
    
    # Calculate mAP
    APs, mAP, TP, FP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, n_classes=21)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return mAP


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_confidence(gt_box_num, pred_box_num, conf):
    """Dynamically adjusts confidence threshold to filter noisy predictions."""
    if pred_box_num > gt_box_num * 3:
        conf += 0.01
    elif pred_box_num < gt_box_num * 2 and conf > 0.01:
        conf -= 0.01
    return conf


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale
    print(f"[*] Adjusted learning rate to: {optimizer.param_groups[0]['lr']:f}")     


if __name__ == '__main__':
    main()