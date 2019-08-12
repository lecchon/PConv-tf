# -*- coding:utf8 -*-

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


class PConv:
    
    
    def __init__(self, image_size = [512,512], mode = tf.estimator.ModeKeys.TRAIN):
        self.is_training = (mode==tf.estimator.ModeKeys.TRAIN)
        ## build model
        self.image = tf.placeholder(dtype = tf.float32, shape = [None,None,None,3], name = "input_image")
        self.mask = tf.placeholder(dtype = tf.float32, shape = [None,None,None,3], name = "input_mask")
        self.reconstr = self.unet(self.image,self.mask)
        self.image_comp = tf.identity(self.image * self.mask + self.reconstr * (1 - self.mask), "output_image")
    
    def unet(self,image,mask):
        '''main network architecture'''
        with tf.variable_scope("unet", reuse = tf.AUTO_REUSE):
            with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
                ## input 512x512
                image_enc1, image_enc1_fea, mask_enc1 = self.downscale(image, mask, 7, 32, "down1")
                ## input 256x256
                image_enc2, image_enc2_fea, mask_enc2 = self.downscale(image_enc1, mask_enc1, 5, 64, "down2")
                ## input 128x128
                image_enc3, image_enc3_fea, mask_enc3 = self.downscale(image_enc2, mask_enc2, 5, 128, "down3")
                ## input 64x64
                image_enc4, image_enc4_fea, mask_enc4 = self.downscale(image_enc3, mask_enc3, 3, 256, "down4")
                ## input 32x32
                image_enc5, image_enc5_fea, mask_enc5 = self.downscale(image_enc4, mask_enc4, 3, 256, "down5")
                ## input 16x16
                image_enc6, image_enc6_fea, mask_enc6 = self.downscale(image_enc5, mask_enc5, 3, 256, "down6")
                ## input 8x8
                image_enc7, image_enc7_fea, mask_enc7 = self.downscale(image_enc6, mask_enc6, 3, 256, "down7")
                ## input 4x4
                image_enc8, image_enc8_fea, mask_enc8 = self.downscale(image_enc7, mask_enc7, 3, 256, "down8")

            with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
                ## input 2x2
                image_dec1,mask_dec1 = self.upscale(image_enc8, image_enc7_fea, mask_enc8, mask_enc7, 3, 256, "up1")
                ## input 4x4
                image_dec2,mask_dec2 = self.upscale(image_dec1, image_enc6_fea, mask_dec1, mask_enc6, 3, 256, "up2")
                ## input 8x8
                image_dec3,mask_dec3 = self.upscale(image_dec2, image_enc5_fea, mask_dec2, mask_enc5, 3, 256, "up3")
                ## input 16x16
                image_dec4,mask_dec4 = self.upscale(image_dec3, image_enc4_fea, mask_dec3, mask_enc4, 3, 256, "up4")
                ## input 32x32
                image_dec5,mask_dec5 = self.upscale(image_dec4, image_enc3_fea, mask_dec4, mask_enc3, 3, 128, "up5")
                ## input 64x64
                image_dec6,mask_dec6 = self.upscale(image_dec5, image_enc2_fea, mask_dec5, mask_enc2, 3, 64, "up6")
                ## input 128x128
                image_dec7,mask_dec7 = self.upscale(image_dec6, image_enc1_fea, mask_dec6, mask_enc1, 3, 32, "up7")
                ## input 256x256
                image_dec8,mask_dec8 = self.upscale(image_dec7, image, mask_dec7, mask, 3, 3, "up8")
                ## input 512x512
                output = tf.contrib.layers.conv2d(image_dec8, 3, 1, activation_fn = tf.nn.sigmoid, scope = "output")
        return output
        
        
    def downscale(self, image_fea, mask_fea, kernel_size, channels, name, use_bias = True, use_bn = True):
        '''down scale operation in U-net'''
        c_in = image_fea.get_shape().as_list()[-1]
        weights_shape = [kernel_size, kernel_size, c_in, channels]
        biases_shape = [channels]
        pad_h = kernel_size//2
        pad_w = kernel_size//2
        image_conv_bn, image_conv, mask_conv = self.pconv_layer(image = image_fea,
                                                                mask = mask_fea,
                                                                weights_shape = weights_shape,
                                                                biases_shape = biases_shape,
                                                                pad_size = (pad_h, pad_w),
                                                                activation_fn = tf.nn.relu,
                                                                name = name,
                                                                use_bias = use_bias)
        return image_conv_bn, image_conv, mask_conv
            
            
    def upscale(self, image_fea, mirror_image_fea, mask_fea, mirror_mask_fea, kernel_size, channels, name, use_bias = True, use_bn = True):
        '''up scale operation in U-net'''
        c_in1 = image_fea.get_shape().as_list()[-1]
        c_in2 = mirror_image_fea.get_shape().as_list()[-1]
        ## concatenate features
        image_fea = tf.keras.layers.UpSampling2D(size=(2,2))(image_fea)
        mask_fea = tf.keras.layers.UpSampling2D(size=(2,2))(mask_fea)
        image_fea = tf.concat([image_fea, mirror_image_fea], axis = -1)
        mask_fea = tf.concat([mask_fea, mirror_mask_fea], axis = -1)
        ## do convolution
        weights_shape = [kernel_size, kernel_size, c_in1+c_in2, channels]
        biases_shape = [channels]
        pad_h = kernel_size//2
        pad_w = kernel_size//2
        image_conv_bn, image_conv, mask_conv = self.pconv_layer(image = image_fea,
                                                                mask = mask_fea,
                                                                weights_shape = weights_shape,
                                                                biases_shape = biases_shape,
                                                                pad_size = (pad_h, pad_w),
                                                                activation_fn = lambda x: tf.nn.leaky_relu(x, 0.2),
                                                                name = name,
                                                                use_bias = use_bias)
        return image_conv,mask_conv
    
    
    def pconv_layer(self, image, mask, weights_shape, biases_shape, pad_size, activation_fn, name, use_bias = True):
        ## get shapes
        pad_h,pad_w = pad_size
        h,w = weights_shape[:2]
        c_in = biases_shape[-1]
        ## initialize weights and biases
        weights = tf.get_variable(name+"_weights", 
                                  shape = weights_shape, 
                                  initializer = tf.contrib.layers.xavier_initializer(), 
                                  dtype = tf.float32,
                                  trainable = True)
        #w1 = tf.get_variable(name+"_depth_weights",
        #                     shape = weights_shape[:-1]+[1],
        #                     initializer = tf.contrib.layers.xavier_initializer(), 
        #                     dtype = tf.float32,
        #                     trainable = True)
        #w2 = tf.get_variable(name+"_point_weights",
        #                     shape = [1,1]+weights_shape[2:],
        #                     initializer = tf.contrib.layers.xavier_initializer(), 
        #                     dtype = tf.float32,
        #                     trainable = True)
        biases = tf.get_variable(name+"_bias",
                                 shape = biases_shape,
                                 initializer = tf.initializers.zeros(),
                                 dtype = tf.float32,
                                 trainable = True)
        weight_for_mask = tf.ones(weights_shape, dtype = tf.float32)
        ## operations
        ### padding
        image_fea_pad = tf.pad(image, [[0,0],[pad_h,pad_h],[pad_h,pad_w],[0,0]])
        mask_fea_pad = tf.pad(mask, [[0,0],[pad_h,pad_h],[pad_h,pad_w],[0,0]])
        masked_image_fea = image_fea_pad * mask_fea_pad
        ### convolutions
        if name.startswith("up"):
            stride = 1
        else:
            stride = 2
        image_conv = tf.nn.convolution(masked_image_fea, weights, strides = (stride,stride), padding = "VALID", name = name+"_img_conv")
        #image_conv = tf.nn.depthwise_conv2d(masked_image_fea, w1, strides = (1, stride, stride, 1), padding = "VALID", name = name+"_depthwise")
        #image_conv = tf.nn.convolution(image_conv, w2, strides = (1,1), padding = "VALID", name = name+"_pointwise")
        mask_conv = tf.nn.convolution(mask_fea_pad, weight_for_mask, strides = (stride,stride), padding = "VALID", name = name + "_msk_conv")
        ratio = h*w/mask_conv
        ratio = tf.where(tf.is_inf(ratio), tf.zeros_like(ratio), ratio) # if value in mask is 0, the division result will be inf
        mask_conv = tf.clip_by_value(mask_conv, 0., 1.)
        image_conv = image_conv * ratio
        if use_bias:
            image_conv = tf.nn.bias_add(image_conv,biases)
        image_conv_bn = self.batch_norm(image_conv, name = name+"_bn")
        return activation_fn(image_conv_bn), image_conv, mask_conv
    
    def batch_norm(self, x, eps = 1e-5, decay = 0.9, name = None):
        with tf.variable_scope(name):
            channels = x.shape.as_list()[-1:]
            moving_mean = tf.get_variable("mean", channels, initializer = tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable("variance", channels, initializer = tf.ones_initializer, trainable=False)
            beta = tf.get_variable("beta", channels, initializer = tf.zeros_initializer)
            gamma = tf.get_variable("gamma", channels, initializer = tf.ones_initializer)
            def mean_var_with_update():
                mean,variance = tf.nn.moments(x, [0,1,2], name="moments")
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)
                                             ]):
                    return tf.identity(mean), tf.identity(variance)
            mean,variance = tf.cond(tf.equal(self.is_training, True), mean_var_with_update, lambda:(moving_mean, moving_variance))
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            