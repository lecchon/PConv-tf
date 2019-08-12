# -*- coding:utf8 -*-

import json
import os
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

class Reader:
    '''
    Reader for model training. Either a configuration json file or manually input parameters supported.
    '''
    def __init__(self, config_file = None, **kwargs):
        self.position = 0
        self.current_epoch = 1
        if config_file is not None:
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = kwargs
        self.num_epoches = int(self.config.get("num_epoches"))
        self.batch_size = int(self.config.get("batch_size"))
        self.image_size = int(self.config.get("image_size"))
        self.data_path = os.path.abspath(self.config.get("data_path"))
        self.img_input = [os.path.join(self.data_path, file_name) for file_name in os.listdir(self.data_path)]
        random.shuffle(self.img_input)
        self.NUM_INPUT_IMAGES = len(self.img_input)

    def next(self):
        img_input = self.img_input[self.position:self.batch_size+self.position]
        ## modify index
        num_input = len(img_input)
        if self.position + self.batch_size >= self.NUM_INPUT_IMAGES:
            self.position = 0
            self.current_epoch += 1
        else:
            self.position = self.position + self.batch_size
        ## read images
        img_list = []
        msk_list = []
        for file_name in img_input:
            img,msk = self._read_image(file_name)
            img_list.append(img)
            msk_list.append(msk)
        return np.array(img_list), np.array(msk_list)
    
    def _read_image(self,x):
        image = Image.open(x).convert("RGB")
        ## resize image
        width, height = image.size
        max_side = max([width,height])
        new_width = int(width/max_side * self.image_size)
        new_height = int(height/max_side * self.image_size)
        image = np.array(image.resize((new_width,new_height))).astype("float32")/255.
        mask = self._gen_mask(new_width,new_height).astype("float32")
        ## pad with zeros
        image_pad = np.zeros((self.image_size,self.image_size,3), dtype = np.float32)
        mask_pad = np.ones((self.image_size,self.image_size,3) , dtype = np.float32)
        top = (self.image_size-new_height)//2
        left = (self.image_size-new_width)//2
        bottom = top + new_height 
        right = left + new_width
        image_pad[top:bottom,left:right,:] = image
        mask_pad[top:bottom,left:right,:] = mask
        return image_pad, mask_pad
    
    def _gen_mask(self, width, height):
        img = np.zeros((height, width, 3), np.uint8)

        # Set size scale
        size = int((width + height) * 0.03)
        
        # Draw random lines
        for _ in range(random.randint(1, 20)):
            x1, x2 = random.randint(1, height), random.randint(1, width)
            y1, y2 = random.randint(1, height), random.randint(1, width)
            thickness = random.randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(random.randint(1, 20)):
            x1, y1 = random.randint(1, height), random.randint(1, width)
            radius = random.randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(random.randint(1, 20)):
            x1, y1 = random.randint(1, height), random.randint(1, width)
            s1, s2 = random.randint(1, height), random.randint(1, width)
            a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
            thickness = random.randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
        return 1-img
    
    def has_next(self):
        return self.current_epoch <= self.num_epoches