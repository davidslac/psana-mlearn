from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## based on this code:

########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import os
import sys
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import h5py

class vgg16:
    def __init__(self, imgs, weights=None, sess=None, trainable=False, stop_at_fc2=False):
        self.imgs = imgs
        self.stop_at_fc2=stop_at_fc2
        self.trainable=trainable
        self.after_relus = []
        self.convlayers(trainable)
        self.fc_layers(trainable)
        if not self.stop_at_fc2:
            self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        self.layer_name_to_op = {'conv1_1':self.conv1_1,
                                 'conv1_2':self.conv1_2,
                                 'pool1':self.pool1,
                                 'conv2_1':self.conv2_1,
                                 'conv2_2':self.conv2_2,
                                 'pool2':self.pool2,
                                 'conv3_1':self.conv3_1,
                                 'conv3_2':self.conv3_2,
                                 'conv3_3':self.conv3_3,
                                 'pool3':self.pool3,
                                 'conv4_1':self.conv4_1,
                                 'conv4_2':self.conv4_2,
                                 'conv4_3':self.conv4_3,
                                 'pool4':self.pool4,
                                 'conv5_1':self.conv5_1,
                                 'conv5_2':self.conv5_2,
                                 'conv5_3':self.conv5_3,
                                 'pool5':self.pool5,
                                 'fc1':self.fc1,
                                 'fc2':self.fc2}
                                 

    def convlayers(self, trainable):
        self.parameters = []

        images = self.imgs

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                 trainable=trainable)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv1_1)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                 trainable=trainable)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv1_2)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                             trainable=trainable)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv2_1)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                                                  trainable=trainable)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv2_2)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                 trainable=trainable)

            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv3_1)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                 trainable=trainable)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv3_2)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights',
                                 trainable=trainable)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv3_3)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv4_1)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv4_2)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv4_3)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv5_1)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv5_2)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=trainable)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.after_relus.append(self.conv5_3)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self, trainable):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=trainable)
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.after_relus.append(self.fc1)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=trainable)
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.after_relus.append(self.fc2)

            self.parameters += [fc2w, fc2b]

        if self.stop_at_fc2:
            return
        
        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=trainable)
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if self.stop_at_fc2 and i >= 30:
                print("hit fc3 - exiting load weights")
                break
            print( i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def get_model_layers(self, sess, imgs, layer_names):
        ops = [self.layer_name_to_op[name] for name in layer_names]
        return sess.run(ops, feed_dict={self.imgs:imgs})
    
def load_image_for_vgg16(dest, img, dbg=False):
    assert dest.shape == (224,224,3)
    assert dest.dtype == np.float32

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if len(img.shape)==2:
        if img.shape != (224, 224):
            img = imresize(img, (224,224))
        for ch in range(3):
            dest[:,:,ch]=img[:,:]

    if len(img.shape)==3:
        assert img.shape[0:2]==(224,224), "load_image_for_vgg16: a 3 channel image must already be sized to 224 224"
        dest[:] = img[:]

    if dbg:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(figsize=(18,12))
        plt.subplot(1,2,1)
        plt.imshow(img, interpolation='none')
        plt.colorbar()
        plt.title('orig img')
        plt.subplot(1,2,2)
        dest2 = dest.copy()
        dest2 /= 256.0
        plt.imshow(dest2, interpolation='none')
        plt.colorbar()
        plt.title('processed img')
        plt.figtext(0.05, 0.95, "psmlearn.load_image_for_vgg16")
        plt.pause(.1)
    return

def create(session, weights):
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    return vgg16(imgs=imgs, weights=weights, sess=session)

def write_codewords(img_iter, h5, dset_name, 
                    weights='vgg16_weights.npz', dbg=False):
    dataReader.loadall(reload=False)
    dataReader.check_processed_means()
    num_samples = dataReader.get_number_of_samples()
    assert (not os.path.exists(output_fname)) or force, "write_codewords: output file %s exists, use --force" % output_fname
    print("psmlearn.vgg16.write_codewords: creating file %s\n  dataReader=%r" % (output_fname, dataReader))
    h5out = h5py.File(output_fname,'w')
    dataReader.copy_to_h5(h5out)
    sess = tf.Session()

    codeword_datasets = {}
    for nm in dataReader.names():
        nm_codewords_1 = np.zeros((num_samples,4096), dtype=np.float32)
        nm_codewords_2 = np.zeros((num_samples,4096), dtype=np.float32)
        for row in range(num_samples):
            nm_img = np.zeros((224,224,3), dtype=np.float32)
            load_image_for_vgg16(nm_img, dataReader.get_image(nm, row), dbg=dbg)
            codewords = sess.run([vgg.fc1, vgg.fc2], feed_dict={vgg.imgs: [nm_img]})
            codeword1, codeword2 = codewords
            nm_codewords_1[row,:] = codeword1[:]
            nm_codewords_2[row,:] = codeword2[:]
            print("nm=%s row=%d" % (nm, row))
            if dbg:
                assert 'q' != raw_input('hit enter or q to quit').strip(), 'quiting'
            sys.stdout.flush()
        nm_ds1 ='%s_codeword1' % nm 
        nm_ds2 ='%s_codeword2' % nm 
        h5out[nm_ds1] = nm_codewords_1
        h5out[nm_ds2] = nm_codewords_2
        codeword_datasets[nm]=(nm_ds1, nm_ds2)
    h5out.close()

    return codeword_datasets
