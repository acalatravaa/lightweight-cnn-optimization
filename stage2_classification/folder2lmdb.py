"""
LMDB Dataset Generator

Converts the PASCAL VOC image dataset into an LMDB (Lightning Memory-Mapped Database) 
format to drastically accelerate I/O operations during the YOLO training loop.
"""

import argparse
import os
import pickle
import shutil
import yaml
from typing import Any, Tuple, List

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.image_augmentation import Image_Augmentation
from data.od_dataset_from_file import DatasetFromFile

# Handle PyTorch version discrepancies for InterpolationMode
if torch.__version__ > '1.8':
    from torchvision.transforms import InterpolationMode
    interp = InterpolationMode.BILINEAR
else:
    interp = 2


class ImageFolderLMDB(data.Dataset):
    """Dataset wrapper for reading from the generated LMDB."""
    
    def __init__(self, db_path: str, transform_size: List[List[int]] = [[352, 352]], phase: str = None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
                             
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
        self.transform_size = transform_size
        self.phase = phase
        self.img_aug = Image_Augmentation()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
            
        unpacked = pickle.loads(byteflow)

        # Decode image
        imgbuf = unpacked[0]
        X_str = np.frombuffer(imgbuf[1], dtype=np.uint8)
        img = cv2.imdecode(X_str, cv2.IMREAD_COLOR)       

        # Process labels and bounding boxes
        target = unpacked[1]
        target2 = torch.Tensor(target)           
        boxes = target2[..., 1:5]

        # Convert relative center coordinates to absolute min/max coordinates
        x1 = (boxes[..., 0] - boxes[..., 2] / 2).unsqueeze(1)
        y1 = (boxes[..., 1] - boxes[..., 3] / 2).unsqueeze(1)
        x2 = (boxes[..., 0] + boxes[..., 2] / 2).unsqueeze(1)
        y2 = (boxes[..., 1] + boxes[..., 3] / 2).unsqueeze(1)
        boxes2 = torch.cat((x1 * img.shape[1], y1 * img.shape[0], x2 * img.shape[1], y2 * img.shape[0]), 1)
        
        labels = target2[..., 0]
        difficulties = torch.zeros_like(labels)
        
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        
        # Apply data augmentations for object detection
        new_img, new_boxes, new_labels, _ = self.img_aug.transform_od(
            image, boxes2, labels, difficulties, 
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], phase=self.phase
        )

        old_dims = torch.FloatTensor([new_img.width, new_img.height, new_img.width, new_img.height]).unsqueeze(0)
        new_boxes2 = new_boxes / old_dims  # Convert back to percentage coordinates
        
        # Convert min/max coordinates back to center coordinates and width/height
        w = (new_boxes2[..., 2] - new_boxes2[..., 0])
        h = (new_boxes2[..., 3] - new_boxes2[..., 1])
        x = (new_boxes2[..., 0] + w / 2).unsqueeze(1)
        y = (new_boxes2[..., 1] + h / 2).unsqueeze(1)
        
        new_boxes2 = torch.cat((x, y, w.unsqueeze(1), h.unsqueeze(1)), 1)
        new_target = torch.cat((new_labels.unsqueeze(1), new_boxes2), 1)

        return new_img, new_target

    def __len__(self) -> int:
        return self.length

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        import random
        images = []
        labels = []
        random_size = random.choice(self.transform_size)
        
        transform = transforms.Compose([
            transforms.Resize(size=random_size, interpolation=interp),
            transforms.ToTensor(),
            self.normalize,
        ])  
        
        for b in batch:
            images.append(transform(b[0]))
            labels.append(b[1])
        
        images = torch.stack(images, dim=0)
        return images, labels  


def generate_lmdb(dataset_yaml_path: str, write_frequency: int = 5000) -> None:
    """Parses the dataset YAML and builds the LMDB structures."""
    dataset_yaml_path = os.path.expanduser(dataset_yaml_path)
    print(f"[*] Loading dataset configurations from {dataset_yaml_path}")

    with open(dataset_yaml_path, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.Loader)
        trainval_config = data["trainval_dataset_path"]
        test_config = data["test_dataset_path"]
  
    trainval_dataset = DatasetFromFile(
        trainval_config['imgs'], trainval_config['annos'], trainval_config['lists'], 
        dataset_name=trainval_config['name'], phase='test', difficultie=False
    )
        
    test_dataset = DatasetFromFile(
        test_config['imgs'], test_config['annos'], test_config['lists'], 
        dataset_name=test_config['name'], phase='test', difficultie=False
    )
    
    outpaths = [trainval_config['lmdb'], test_config['lmdb']]
    datasets_to_process = [trainval_dataset, test_dataset]
    
    for i, dataset in enumerate(datasets_to_process):        
        data_loader = DataLoader(dataset, num_workers=4, collate_fn=lambda x: x)
        lmdb_path = os.path.expanduser(outpaths[i])
        
        if os.path.exists(lmdb_path) and os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
            
        os.makedirs(lmdb_path)
        print(f"[*] Generating LMDB structure at: {lmdb_path}")
        
        db = lmdb.open(lmdb_path, subdir=True, map_size=1099511627776 * 2, 
                       readonly=False, meminit=False, map_async=True)

        txn = db.begin(write=True)
        total_boxes = 0
        
        for idx, batch in enumerate(data_loader):
            image, label = batch[0][0], batch[0][1]
            total_boxes += len(label)
            txn.put(f'{idx}'.encode('ascii'), pickle.dumps((image, label)))
            
            if idx % write_frequency == 0 and idx > 0:
                print(f"    Processed [{idx}/{len(data_loader)}]")
                txn.commit()
                txn = db.begin(write=True)

        print(f"[*] Total bounding boxes processed: {total_boxes}")
        txn.commit()
        
        keys = [f'{k}'.encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))

        print("[*] Flushing database to disk...")
        db.sync()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC to LMDB")
    parser.add_argument("-d", "--dataset", help="Path to VOC dataset config YAML", default='data/voc_data.yaml')
    args = parser.parse_args()
    generate_lmdb(args.dataset)