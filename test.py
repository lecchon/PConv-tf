# -*- coding:utf8 -*-

from PIL import Image
import numpy as np
import tensorflow as tf

import sys
import os
import time
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(".")

from model import PConv

from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_io,graph_util
from tensorflow.python.platform import gfile

def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')
    
    parser.add_argument(
        '-image_size', '--image_size',
        type=int, default=512,
        help='input image size'
    )
    parser.add_argument(
        '-data_path', '--data_path',
        type=str, default=None,
        help='path of input images'
    )
    parser.add_argument(
        '-model_file', '--model_file',
        type=str, default="./model",
        help='directory for saving models'
    )
    
    return parser.parse_args()


def resize_and_pad(x,y,size):
    w,h = x.size
    w_new = int(size/max([w,h])*w)
    h_new = int(size/max([w,h])*h)
    x_ = x.resize((w_new,h_new))
    y_ = y.resize((w_new,h_new))
    x_ = np.array(x_)
    y_ = np.array(y_)
    new_x = np.zeros((size,size,3))
    new_y = np.zeros((size,size,3))
    new_x[(size-h_new)//2:(size-h_new)//2+h_new,(size-w_new)//2:(size-w_new)//2+w_new,:] = x_
    new_y[(size-h_new)//2:(size-h_new)//2+h_new,(size-w_new)//2:(size-w_new)//2+w_new,:] = y_
    return (new_x/255.).astype("float32"),(new_y/255.).astype("float32")


def main():
    args = parse_args()
    model = PConv(image_size = [args.image_size,args.image_size], mode = "test")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    saver = tf.train.Saver()
    ## load ckpt
    saver.restore(session, args.model_file)
    flag = True
    while flag:
        input_image = input("> input your image and mask: ")
        if input_image == "q":
            flag = False
            print("Exiting... Bye!")
        if not os.path.exist(input_image):
            print("[！] `%s` doesn't exsits! Please retry!" % input_image)
        elif not input_image.startswith("image_"):
            print("[！] Please rename your input image with beginning `image_` as well as beginning `mask_` for mask map.")
        else:
            mask_image = input_image.replace("image_","mask_")
            if not os.path.exist(mask_image):
                print("[！] `%s` doesn't exsits! Please retry!" % mask_image)
            else:
                input_img = Image.open(input_image).convert("RGB")
                input_msk = Image.open(mask_image).convert("L")
                input_img_val,input_msk_val,loss_msk_val = resize_and_pad(input_img, input_msk, args.image_size)
                output_val = session.run(model.image_comp, feed_dict = {model.image:input_img_val[np.newaxis,...],
                                                                        model.mask:input_img_val[np.newaxis,...],})
                Image.fromarray((output_val[0] * 255).astype("uint8")).save(os.path.basename(input_image).replace("image_", "test"))

if __name__ == "__main__":
    main()