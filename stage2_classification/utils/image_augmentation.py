"""
Image Augmentation Pipeline for Object Detection
"""

import random
import torch
import torchvision.transforms.functional as FT

from utils.iou import find_jaccard_overlap


class Image_Augmentation:
    def expand_od(self, image: torch.Tensor, boxes: torch.Tensor, filler: list):
        original_h, original_w = image.size(1), image.size(2)
        scale = random.uniform(1, 4)
        new_h, new_w = int(scale * original_h), int(scale * original_w)

        filler_t = torch.FloatTensor(filler)
        new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler_t.unsqueeze(1).unsqueeze(1)

        left = random.randint(0, new_w - original_w)
        top = random.randint(0, new_h - original_h)
        new_image[:, top:top + original_h, left:left + original_w] = image
        new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

        return new_image, new_boxes

    def random_crop_od(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, difficulties: torch.Tensor):
        original_h, original_w = image.size(1), image.size(2)
        while True:
            min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])
            if min_overlap is None:
                return image, boxes, labels, difficulties

            for _ in range(50):
                scale_h, scale_w = random.uniform(0.3, 1), random.uniform(0.3, 1)
                new_h, new_w = int(scale_h * original_h), int(scale_w * original_w)

                if not 0.5 < (new_h / new_w) < 2:
                    continue

                left = random.randint(0, original_w - new_w)
                top = random.randint(0, original_h - new_h)
                crop = torch.FloatTensor([left, top, left + new_w, top + new_h])

                overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)
                if overlap.max().item() < min_overlap:
                    continue

                new_image = image[:, top:top + new_h, left:left + new_w]
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
                
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < left + new_w) * \
                                  (bb_centers[:, 1] > top) * (bb_centers[:, 1] < top + new_h)

                if not centers_in_crop.any():
                    continue

                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2]) - crop[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:]) - crop[:2]

                return new_image, new_boxes, new_labels, new_difficulties

    def flip_od(self, image: torch.Tensor, boxes: torch.Tensor):
        new_image = FT.hflip(image)
        new_boxes = boxes.clone()
        new_boxes[:, 0] = image.width - boxes[:, 0] - 1
        new_boxes[:, 2] = image.width - boxes[:, 2] - 1
        new_boxes = new_boxes[:, [2, 1, 0, 3]]
        return new_image, new_boxes

    def photometric_distort(self, image: torch.Tensor):
        distortions = [
            FT.adjust_brightness, FT.adjust_contrast,
            FT.adjust_saturation, FT.adjust_hue, FT.adjust_gamma
        ]
        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                # Corrected Python operator for string comparison
                if d.__name__ == 'adjust_hue':
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    adjust_factor = random.uniform(0.5, 1.5)
                image = d(image, adjust_factor)
        return image

    def transform_od(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, difficulties: torch.Tensor, 
                     mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], phase: str = 'train'):
        assert phase in {'train', 'test'}

        if phase == 'train':
            image = self.photometric_distort(image)
            image = FT.to_tensor(image)
            
            if random.random() < 0.5:
                image, boxes = self.expand_od(image, boxes, filler=mean)

            image, boxes, labels, difficulties = self.random_crop_od(image, boxes, labels, difficulties)
            image = FT.to_pil_image(image)

            if random.random() < 0.5:
                image, boxes = self.flip_od(image, boxes)

        return image, boxes, labels, difficulties