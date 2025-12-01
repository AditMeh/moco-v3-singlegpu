# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf


class SharedInitialCrop:
    """Crop image to a random square patch of size patch_size*2, shared between both augmentations"""
    
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.crop_size = patch_size * 2
    
    def __call__(self, img):
        """
        Crop image to a random square of size patch_size*2.
        If crop_size is bigger than image dimension, don't crop that dimension.
        """
        width, height = img.size
        crop_size = min(self.crop_size, width, height)
        
        # Random top-left corner
        top = random.randint(0, max(0, height - crop_size))
        left = random.randint(0, max(0, width - crop_size))
        
        # Crop to square
        img = img.crop((left, top, left + crop_size, top + crop_size))
        return img


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, shared_initial_crop=None):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.shared_initial_crop = shared_initial_crop

    def __call__(self, x):
        # Apply shared initial crop if specified
        if self.shared_initial_crop is not None:
            x = self.shared_initial_crop(x)
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)