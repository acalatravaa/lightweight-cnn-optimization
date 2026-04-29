"""
YOLO Object Detection Inference Script

Loads a trained MobileNetV2-YOLO architecture and evaluates it against
raw images, visualizing the predicted bounding boxes and class labels.
"""

import argparse
import os
import yaml

import cv2
import filetype
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from models.voc.mbv2_yolo import yolo


# PASCAL VOC Distinct Visual Colors
DISTINCT_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', 
                   '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080']       
    
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')    


def load_model_weights(model: torch.nn.Module, weight_path: str, device: torch.device) -> torch.nn.Module:
    """Safely loads checkpoint weights, matching keys to the current architecture."""
    checkpoint = torch.load(weight_path, map_location=device)
    pretrained_dict = checkpoint.get('model', checkpoint)
    
    model_dict = model.state_dict()
    # Filter out unnecessary or mismatched keys (vital for evaluating decoupled heads)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    if not pretrained_dict:
        raise ValueError(f"Failed to load weights from {weight_path}. No matching keys found.")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"[*] Loaded trained weights from: {weight_path}")
    return model  


def execute_inference(model: torch.nn.Module, original_image: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocesses the image and executes the forward pass."""
    transform_test = transforms.Compose([
        transforms.Resize(size=(416, 416), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image_tensor = transform_test(original_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        detections = model(image_tensor)
        
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO Inference Visualization')
    parser.add_argument('-c', '--checkpoint', required=True, type=str, help='Path to load checkpoint .pth.tar')
    parser.add_argument('-y', '--yaml', default='models/voc/config.yaml', type=str, help='Path to model config')                     
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to input image file') 
    args = parser.parse_args()

    if not os.path.isfile(args.yaml):
        raise FileNotFoundError(f"Configuration file missing: {args.yaml}")
        
    with open(args.yaml, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)      

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Executing inference on device: {device}")
    
    # Initialize and load model
    model = yolo(config=config)
    model = load_model_weights(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Set confidence thresholds for visualization
    model.yolo_losses[0].val_conf = 0.01 
    model.yolo_losses[1].val_conf = 0.01 

    # Verify input file type
    kind = filetype.guess(args.input)
    if kind is None or kind.extension not in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
        raise ValueError(f"Input file {args.input} is not a valid image format.")

    original_image = Image.open(args.input, mode='r').convert('RGB')
    height, width = np.asarray(original_image).shape[:2]
    
    det_boxes = execute_inference(model, original_image, device)

    # Visualization Setup
    draw = ImageDraw.Draw(original_image)     
    try:
        font = ImageFont.truetype("data/Arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    if det_boxes is not None:
        for bbox in det_boxes[0]:
            box_location = bbox[:4].tolist()
            conf = bbox[4].item()
            cls_conf = bbox[5].item()
            cls_index = int(bbox[6].item())
            
            # Draw threshold
            if conf * cls_conf > 0.15:
                # Scale coordinates back to original image dimensions
                box_location[0] *= width
                box_location[1] *= height
                box_location[2] *= width
                box_location[3] *= height  
                
                draw.rectangle(xy=box_location, outline=DISTINCT_COLORS[cls_index % len(DISTINCT_COLORS)], width=3)
                 
                text_size = 15
                text_location = [box_location[0] + 3., box_location[1] - text_size]
                draw.text(xy=text_location, text=CLASSES[cls_index].lower(), fill='red', font=font)  

    output_path = f"result_{os.path.basename(args.input)}"
    original_image.save(output_path)
    print(f"[*] Visualized result saved successfully to: {output_path}")

  
if __name__ == '__main__':
    main()