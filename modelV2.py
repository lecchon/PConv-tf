# -*- coding:utf8 -*-

import tensorflow as tf
slim = tf.contrib.slim


class PConv:
    
    
    def __init__(self, image_size = [512,512]):
        ## build model
        self.mode = tf.placeholder(dtype=tf.int32, name="training_phrase", shape=[])
        self.image = tf.placeholder(dtype = tf.float32, shape = [None]+image_size+[3], name = "input_image")
        self.mask = tf.placeholder(dtype = tf.float32, shape = [None]+image_size+[3], name = "input_mask")
        self.reconstr = self.unet(self.image,self.mask)
        self.image_comp = tf.identity(self.image * self.mask + self.reconstr * (1 - self.mask), "output_image")
        
    
    def unet(self,image,mask):
        '''main network architecture'''
        with tf.variable_scope("unet", reuse = tf.AUTO_REUSE):
            with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
                image = image * mask
                net = tf.concat([image, mask], axis = -1)
                cnum = 32
                net, fea1 = self.gated_conv(net, cnum*1, 3, 1, 2, name="conv1") # down x2: 256 NOTICE: ksize 5 -> 3
                net, fea2 = self.gated_conv(net, cnum*1, 3, 1, 2, name="conv2") # down x4: 128
                net, fea3 = self.gated_conv(net, cnum*2, 3, 1, 2, name="conv3") # down x16: 64
                net, fea4 = self.gated_conv(net, cnum*2, 3, 1, 2, name="conv4") # down x32: 32
                net, fea5 = self.gated_conv(net, cnum*4, 3, 1, 2, name="conv5") # down x64: 16
                net, fea6 = self.gated_conv(net, cnum*4, 3, 1, 2, name="conv6") # down x128: 8
                net, fea7 = self.gated_conv(net, cnum*4, 3, 1, 2, name="conv7") # NOTICE: stride 5 -> 3
                net, fea8 = self.gated_conv(net, cnum*4, 3, 1, 2, name="conv8") # NOTICE: stride 5 -> 3
            with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
                net = self.gated_deconv(fea8, fea7, cnum*4, 3, 1, 2, name="deconv1")# NOTICE: stride 5 -> 3
                net = self.gated_deconv(net, fea6, cnum*4, 3, 1, 2, name="deconv2") # NOTICE: stride 5 -> 3
                net = self.gated_deconv(net, fea5, cnum*4, 3, 1, 2, name="deconv3") # up x2: 16
                net = self.gated_deconv(net, fea4, cnum*4, 3, 1, 2, name="deconv4") # up x4: 32
                net = self.gated_deconv(net, fea3, cnum*2, 3, 1, 2, name="deconv5") # up x8: 64
                net = self.gated_deconv(net, fea2, cnum*2, 3, 1, 2, name="deconv6") # up x16: 128
                net = self.gated_deconv(net, fea1, cnum*1, 3, 1, 2, name="deconv7") # up x32: 256
                net = self.gated_deconv(net, image, cnum*1, 3, 1, 2, name="deconv8") # up x64: 512
            net = slim.conv2d(net, 3, 1, activation_fn=None, scope="logit")
            net = tf.clip_by_value(net, 0.0, 1.0, name="output")
        return net
        
        
    def gated_conv(self, x, filters, ksize, rate, stride, activation=tf.nn.relu, name=None):
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0],[p,p],[p,p],[0,0]])
        k = slim.conv2d(x, filters, ksize, stride, padding="valid", activation_fn=activation, scope=name+"_k")
        q = slim.conv2d(x, filters, ksize, stride, padding="valid", activation_fn=tf.nn.sigmoid, scope=name+"_q")
        v = q*k
        return v,v
    
    def gated_deconv(self, x, y, filters, ksize, rate, stride, activation=tf.nn.relu, name=None):
        x = tf.keras.layers.UpSampling2D(size=(stride,stride))(x)
        x = tf.concat([x,y],axis=-1)
        x,_ = self.gated_conv(x, filters, ksize, rate, stride=1, activation=activation, name=name)
        return x
    
    def self_attention(self, x):
        pass