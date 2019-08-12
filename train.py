# -*- coding:utf8 -*-

import tensorflow as tf

import sys
import os
import time
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(".")

from loss import Loss
from reader import Reader
from model import PConv

from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_io,graph_util
from tensorflow.python.platform import gfile

def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')
    parser.add_argument(
        '-num_epoch', '--num_epoch',
        type=int, default=100,
        help='training epoches.'
    )
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=4,
        help='batch size per iteration.'
    )
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
        '-learning_rate', '--learning_rate',
        type=float, default=0.0002,
        help='learning rate for training'
    )
    parser.add_argument(
        '-model_path', '--model_path',
        type=str, default="./model",
        help='directory for saving models'
    )
    parser.add_argument(
        '-mode', '--mode',
        type=str, default="train",
        help='training from sketch or fine-tuning',
        choices = ["train","fine-tune"]
    )
    parser.add_argument(
        '-config', '--config',
        type=str, default=None,
        help='json configuration file',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    t0 = time.time()
    ## define some variables
    global_step = tf.get_variable(name = "global_step", dtype = tf.int32, shape = [], trainable = False)
    ## initialize reader.Either using configuration json file or assigning arguments.
    if args.config is None:
        if args.data_path is None:
            raise Exception("Either configuration file or data path should be assigned!")
        reader = Reader(**{"num_epoch":args.num_epoch, "batch_size":args.batch_size, "image_size":args.image_size, "data_path":args.data_path, })
    else:
        reader = Reader(config_file = args.config)
    ## gpu configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    
    ## build model
    model = PConv(image_size = [reader.image_size,reader.image_size])
    ## compute loss
    loss = Loss(model.image, model.mask, model.reconstr, model.image_comp, reader.image_size)
    loss_ = loss.total_loss()
    ## training ops
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_, var_list = tf.trainable_variables(), global_step=global_step)
    
    ## initialize variables
    var_list = [ t for t in tf.global_variables() if not t.name.startswith("vgg16")]
    session.run(tf.variables_initializer(var_list))
    saver = tf.train.Saver(var_list)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if args.mode == "fine-tune":
        ### load pretrained model
        ckpt = tf.train.latest_checkpoint(args.model_path)
        if ckpt is not None:
            saver.restore(session, ckpt)
        else:
            print("[WARN] %s is empty or cannot find latest checkpoint file." % args.model_path)
    
    ## add summary
    tf.summary.image("input_image", loss.image_in)
    tf.summary.image("input_mask", loss.mask_in)
    tf.summary.image("image_comp", loss.image_comp)
    tf.summary.image("image_recontr", loss.image_out)
    tf.summary.scalar("hole_loss", loss.l1, family = "image_loss")
    tf.summary.scalar("valid_loss", loss.l2, family = "image_loss")
    tf.summary.scalar("perceptual_loss", loss.l3, family = "vgg_loss")
    tf.summary.scalar("style_loss", loss.l4, family = "vgg_loss")
    tf.summary.scalar("tv_loss", loss.l5, family = "tv_loss")
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log", session.graph)
    run_metadata = tf.RunMetadata()
    
    ## start training
    epoch = reader.current_epoch
    while(reader.has_next()):
        image_val,mask_val = reader.next()
        _, summary, loss_val, step = session.run([train_op, merged, loss_, global_step], feed_dict = {model.image:image_val, model.mask:mask_val})
        if step % 50 == 0:
            writer.add_run_metadata(run_metadata, 'step%d' % step)
            writer.add_summary(summary,step)
            print("Epoch:%s\tLoss:%s\tSpends:%.2fs" % (reader.current_epoch,loss_val,time.time()-t0))
        if step % 10000 == 0:
            saver.save(session, os.path.join(args.model_path, "model.ckpt"), step)
        ## convert to pb file at the end of every epoch
        if reader.current_epoch > epoch:
            constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [model.image_comp.op.name])
            transformed_graph_def = TransformGraph(constant_graph,
                                           [model.image.op.name,model.mask.op.name],
                                           [model.image_comp.op.name],
                                           ["add_default_attributes",
                                            "remove_nodes(op=Identity,op=CheckNumerics)",
                                            "fold_constants(ignore_errors=true)",
                                            "fold_batch_norms",
                                            "fold_old_batch_norms",])
            with tf.gfile.FastGFile("model-%s.pb" % epoch, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            epoch = reader.current_epoch
        ## save final model
        saver.save(session, os.path.join(args.model_path, "model.ckpt"), step)
        
if __name__ == "__main__":
    main()