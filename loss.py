# -*- coding:utf8 -*-

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg


class Loss:
    
    def __init__(self, ground_true, mask, reconstruction, comp, image_size = 512, weights = [1,6,0.05,120,0.1], loss_layers = ["block1_pool","block2_pool","block3_pool"]):
        self.image_in = ground_true
        self.mask_in = mask
        self.image_out = reconstruction
        self.image_comp = comp
        self.weigths = weights
        ## load vgg16
        self.mean = tf.convert_to_tensor([0.485, 0.456, 0.406],dtype=tf.float32,name="mean")
        self.std = tf.convert_to_tensor([0.229, 0.224, 0.225],dtype=tf.float32,name="std")
        inputs = tf.keras.layers.Input((image_size,image_size,3),dtype=tf.float32)
        processed = tf.keras.layers.Lambda(lambda x: (x-self.mean) / self.std)(inputs)
        vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=processed)
        outputs = [vgg16.get_layer(l).output for l in loss_layers]
        self.loss_vgg = tf.keras.models.Model(inputs, outputs)
        self.loss_vgg.trainable = False
        ## get vgg layers output
        self.phi_out = self.loss_vgg(self.image_out)#.predict(self.image_out,steps=1)
        self.phi_in = self.loss_vgg(self.image_in)#.predict(self.image_in,steps=1)
        self.phi_comp = self.loss_vgg(self.image_comp)#.predict(self.image_comp,steps=1)
    
    def total_loss(self):
        ## compute losses
        self.l1 = self.weigths[0]*self.valid_loss()
        self.l2 = self.weigths[1]*self.hole_loss()
        self.l3 = self.weigths[2]*self.perceptual_loss()
        self.l4 = self.weigths[3]*self.style_loss()
        self.l5 = self.weigths[4]*self.tv_loss()
        return self.l1 + self.l2 + self.l3 + self.l4 + self.l5
    
    def hole_loss(self):
        return self.l1_loss((1-self.mask_in)*self.image_out, (1-self.mask_in)*self.image_in)
    
    def valid_loss(self):
        return self.l1_loss(self.mask_in*self.image_out, self.mask_in*self.image_in)

    def perceptual_loss(self):
        diff1 = []
        diff2 = []
        for i,o,c in zip(self.phi_in, self.phi_out, self.phi_comp):
            diff1.append(self.l1_loss(i,o))
            diff2.append(self.l1_loss(i,c))
        return tf.add_n(diff1) + tf.add_n(diff2)
        
    def style_loss(self):
        diff1 = []
        diff2 = []
        for i,o,c in zip(self.phi_in, self.phi_out, self.phi_comp):
            diff1.append(self.l1_loss(self.gram(i),self.gram(o)))
            diff2.append(self.l1_loss(self.gram(i),self.gram(c)))
        return tf.add_n(diff1) + tf.add_n(diff2)
    
    def tv_loss(self):
        return tf.reduce_mean(tf.image.total_variation(self.image_comp*(1-self.mask_in))/tf.reduce_sum(1-self.mask_in, axis=[1,2,3]))
        
    def l1_loss(self,x,y):
        return tf.reduce_mean(tf.reduce_mean(tf.abs(x-y), axis = list(range(1,x.shape.rank))))
    
    def gram(self,x):
        c = x.shape.as_list()[-1]
        x_ = tf.reshape(x, tf.stack([-1, tf.shape(x)[1]*tf.shape(x)[2], c]))
        return tf.matmul(x_, x_, transpose_a=True)/ tf.to_float(tf.shape(x)[1]*tf.shape(x)[2]*c)