import tensorflow as tf
tf.enable_eager_execution()

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
from tensorflow.keras import layers
import time
import pandas as pd
import cv2
import xml.etree.ElementTree as ET

from IPython import display

WIDTH = 64
NUM_CHANNELS = 3
BUFFER_SIZE = 25000
BATCH_SIZE = 64
WEIGHT_INIT_STDDEV = 0.2

EPOCHS = 5
IMAGE_PATH = "../input/all-dogs/all-dogs/*"
ANNOTATION_PATH = "../input/annotation/Annotation/*/*"

# Generate the table of bounding boxes, whose keys are the filename IDs for each dog
# image / CSV.
# annots: The list of annotation filenames - each a CSV corresponding to a dog image
# with relevant information such as bounding boxes for the different dogs.
def getBBTable(annots):
    bb_tbl = {}
    num_bbs = 0
    for a in annots:
        et = ET.parse(a)
        root = et.getroot()
        objects = root.findall("object")
        bbs = []
        for o in objects:
            bb = o.find("bndbox")
            xmin = int(bb.find("xmin").text)
            xmax = int(bb.find("xmax").text)
            ymin = int(bb.find("ymin").text)
            ymax = int(bb.find("ymax").text)
            
            # If the image is not square, crop it.
            
            width = np.min((xmax - xmin, ymax - ymin))
            bbs.append((xmin, ymin, xmin + width, ymin + width)) 
            
            num_bbs += 1
        fid = a.split("/")[-1]
        bb_tbl[fid] = bbs
    return (bb_tbl, num_bbs)

# Generate the image input tensor from the image data, performing slight augmentation
# by flipping each image vertically.
# imgs: The list of dog image filenames.
# bb_tbl: The table of bounding boxes.
# num_bbs: The total number of bounding boxes in our dataset.
# width: The resized size of each image in our dataset along both axes.
# channels: The number of channels for our images.
def getInputTensor(imgs, bb_tbl, num_bbs, width, channels):
    tensor = np.zeros((2*num_bbs, width, width, channels))
    tensor = np.float32(tensor)
    idx = 0
    for img_file in imgs:
        fid = img_file.split("/")[-1][:-4]
        img = Image.open(img_file)
        for bb in bb_tbl[fid]:
            cimg = img.crop(bb)
            cimg = cimg.resize((width, width), Image.ANTIALIAS)
            cimg_flip = cimg.transpose(Image.FLIP_LEFT_RIGHT)
            tensor[idx, :, :, :] = (np.asarray(cimg) - 127.5) / 127.5 # Normalize to [-1, 1].
            tensor[idx+1, :, :, :] = (np.asarray(cimg_flip) - 127.5) / 127.5
            assert(np.any(tensor[idx, :, :, :] != tensor[idx+1, :, :, :]))
            idx += 2
    return tensor

# Get the generator model - a dense layer with ReLU activiation followed by several
# deconvolutional layers with batch normalization and ReLU activation, ending with
# one final deconvolutional layer with tanh activiation.
# channels: The number of image channels.
def getGenerator(channels):
    model = tf.keras.Sequential()
    
    # 100 x 1 -> 4x4x512
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 512)))
    
    # 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x32
    for i in range(4):
        n_f = 2**(8 - i)
        model.add(layers.Conv2DTranspose(n_f, (5, 5), strides=(2, 2), padding="same", use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
    # 64x64x32 -> 64x64x3
    model.add(layers.Conv2DTranspose(channels, (5, 5), strides=(1, 1), padding="same", use_bias=False,
        activation="tanh", kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)))
    
    return model

# Get the discriminator model.  Several convolutional layers with batch normalization
# and ReLU activation, followed by one final dense layer with no activation (thus
# returning a logit).
# channels: The number of image channels.
def getDiscriminator(channels):
    model = tf.keras.Sequential()
    
    # 64x64x3 -> 32x32x32
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=(64, 64, channels),
        kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # 32x32x32 -> 16x16x64 -> 8x8x128 -> 8x8x256 -> 4x4x512
    num_filters_list = [64, 128, 256, 512]
    stride_list = [2, 2, 1, 2]
    
    for (n_f, s) in zip(num_filters_list, stride_list):
        model.add(layers.Conv2D(n_f, (5, 5), strides=(s, s), padding="same",
            kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function - binary cross entropy using slightly noisy labels
# and the discriminator's output for the generated images and the true images.
# gen_out: The output labels of generated images, according to the discriminator.
# true_out: The output labels of true images, according to the discriminator.
def discriminatorLoss(gen_out, true_out):
    gen_labels = tf.random.uniform(gen_out.shape, 0, 0.3)
    gen_loss = cross_entropy(gen_labels, gen_out)
    
    true_labels = tf.random.uniform(true_out.shape, 0.7, 1.2)
    true_loss = cross_entropy(true_labels, true_out)
    return gen_loss + true_loss

# Generator loss function - binary cross entropy using noisy labels and the
# discriminator's output for the generated images.
# gen_out: The output labels of generated images, according to the discriminator.
def generatorLoss(gen_out):
    gen_labels = tf.random.uniform(gen_out.shape, 0.7, 1.2)
    return cross_entropy(gen_labels, gen_out)

# Perform a training step on a single batch.
# batch: The image batch with which training should be performed.
# gen: The generator model.
# disc: The discriminator model.
# opt_gen: The generator optimizer.
# opt_disc: The discriminator optimizer.
def trainingStep(batch, gen, disc, opt_gen, opt_disc):
    z = tf.random.uniform((batch.shape[0], 100), -1, 1)
    noise = tf.random.normal(batch.shape, mean=0.0, stddev=np.random.uniform(0.0, 0.1))
    noisy_batch = batch + noise
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_imgs = gen(z, training=True)
        
        gen_out = disc(gen_imgs, training=True)
        true_out = disc(noisy_batch, training=True)
        
        gen_loss = generatorLoss(gen_out)
        disc_loss = discriminatorLoss(gen_out, true_out)
    
    grad_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)
    grad_disc = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    opt_gen.apply_gradients(zip(grad_gen, gen.trainable_variables))
    opt_disc.apply_gradients(zip(grad_disc, disc.trainable_variables))
    
    return gen_loss, disc_loss

# Perform training for the specified number of epochs.
# ds: The full, already prepared image dataset.
# gen: The generator model.
# disc: The discriminator model.
# opt_gen: The generator optimizer.
# opt_disc: The discriminator optimizer.
# num_epochs: The total number of epochs of training.
def train(ds, gen, disc, opt_gen, opt_disc, num_epochs):
    z = tf.random.uniform((5*5, 100), -1, 1)
    
    epochs = []
    gen_losses = []
    disc_losses = []
    
    for epoch in range(num_epochs):
        print("EPOCH " + str(epoch))
        
        gen_losses_per_batch = []
        disc_losses_per_batch = []
        
        if (epoch + 1) % 5 == 0:
            showGeneratedImages(gen, z, epoch, 5, 5)
        for batch in ds:
            (gen_loss, disc_loss) = trainingStep(batch, gen, disc, opt_gen, opt_disc)
            gen_losses_per_batch.append(gen_loss)
            disc_losses_per_batch.append(disc_loss)
        
        epochs.append(epoch)
        gen_losses.append(np.mean(gen_losses_per_batch))
        disc_losses.append(np.mean(disc_losses_per_batch))
    
    f, ax = plt.subplots()
    ax.plot(epochs, gen_losses, "r", label="Generator Loss")
    ax.plot(epochs, disc_losses, "b", label="Discriminator Loss")
    ax.set_title("Losses With Respect to Number of Epochs")
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    plt.savefig("LOSS_GRAPH.png")

# Show a set of generated images using a fixed noise vector for each.
# gen: The generator model.
# noise: The array of noise vectors to be used.  First dimension should be width
# times height.
# width: The number of images to be displayed in the x-direction.
# height: The number of images to be displayed in the y-direction.
def showGeneratedImages(gen, z, epoch, width, height):
    gen_imgs = gen(z, training=False)
    
    f, axarr = plt.subplots(width, height)
    
    for i in range(width):
        for j in range(height):
            axarr[i, j].imshow(gen_imgs[i*width + j, :, :, :])
    
    plt.axis("off")
    plt.savefig("IMGS_EPOCH_" + str(epoch) + ".png")

imgs = glob.glob(IMAGE_PATH)
annots = glob.glob(ANNOTATION_PATH)

(bb_tbl, num_bbs) = getBBTable(annots)
tensor = getInputTensor(imgs, bb_tbl, num_bbs, WIDTH, NUM_CHANNELS)

tensor = tf.cast(tensor, "float32")

ds = tf.data.Dataset.from_tensor_slices(tensor).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen = getGenerator(NUM_CHANNELS)
disc = getDiscriminator(NUM_CHANNELS)

opt_gen = tf.train.AdamOptimizer(learning_rate = 1e-4)
opt_disc = tf.train.AdamOptimizer(learning_rate = 1e-4)

train(ds, gen, disc, opt_gen, opt_disc, EPOCHS)
