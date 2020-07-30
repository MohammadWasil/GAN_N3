from google.colab import auth
from googleapiclient.discovery import build
from google.colab import drive

import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from numpy.random import randn
from numpy.random import randint
import matplotlib.pyplot as plt



def folder_download(folder_id):
  # authenticate
  
  auth.authenticate_user()
  # get folder_name  
  service = build('drive', 'v3')
  folder_name = service.files().get(fileId=folder_id).execute()['name']
  # import library and download
  #!wget -qnc https://github.com/segnolin/google-drive-folder-downloader/raw/master/download.py
  
  from download import download_folder
  download_folder(service, folder_id, './', folder_name)
  return folder_name

def Data_Generation(args, df):
      
    IMAGE_CHANNELS = 3

    targets = ["Eyeglasses", "Rosy_Cheeks", "Goatee"]

    for col in targets:
        examplesize = 12000
        df_col = df[df[col]==1]
        
        #Only sample examplesize many when there are actually that many instances
        if(df_col["filename"].shape[0] >= examplesize):
            listdir_x = list(df_col["filename"].sample(examplesize, replace=False, random_state=1337))
        else:
            listdir_x = list(df_col["filename"])
            examplesize = len(listdir_x)
            
        print(f"Attribute {col} found {examplesize} examples.")

        training_binary_path = os.path.join(args.NUMPY_FILES,
                f'training_data_{args.IMAGE_SIZE}_{args.IMAGE_SIZE}_{col}_{examplesize}.npy')

        print(f"Looking for file: {training_binary_path}")

        if not os.path.isfile(training_binary_path):
            start = time.time()
            print("Loading training images...")

            training_data = []
            faces_path = args.DATA_PATH

            for filename in tqdm(listdir_x):
                path = os.path.join(faces_path,filename)
                image = Image.open(path).resize((args.IMAGE_SIZE, args.IMAGE_SIZE),Image.ANTIALIAS)
                training_data.append(np.asarray(image))
            
            training_data = np.reshape(training_data,(-1, args.IMAGE_SIZE, args.IMAGE_SIZE, IMAGE_CHANNELS))
            # float16 saves some space instead of float32
            training_data = training_data.astype(np.float16)
            
            # rescale to [-1,1]
            training_data = (training_data/127.5) - 1

            print("Saving training image binary...")
            np.save(training_binary_path, training_data)
            elapsed = time.time()-start
            print (f'Image preprocess time:',elapsed)
        else:
            print("Loading previous training pickle...")
            training_data = np.load(training_binary_path)

def Load_Data(args):
  """
  Load_Data: Load the image and labels fro training.

  Arguments:
        args: parser which contains all the variables and paths.

  Returns:
        X: Input trainng images.
        y: Input training labels.
  """
  # This would be the class header for 40 classes
  #CLASS_HEADER = list(df.columns)[1:]

  # For this experiment we only have 3 classes
  CLASS_HEADER = ["Eyeglasses", "Rosy_Cheeks", "Goatee"]

  # Load the Data - 3 classes

  # needs to be done to initialze the dimension
  X = np.load(os.path.join(args.NUMPY_FILES , "training_data_128_128_Eyeglasses_12000.npy"))
  y = np.array([0]*12000)
  ii=1

  for klasse in CLASS_HEADER[1:]:
      X = np.concatenate([X, np.load(os.path.join(args.NUMPY_FILES, f"training_data_128_128_{klasse}_12000.npy"))])
      y = np.concatenate([y, np.array([ii]*12000)])
      ii=ii+1

  return X, y



# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# select real samples
def generate_real_samples(X, y, n_samples):
    # choose random instances
    ix = randint(0, X.shape[0], n_samples)
    # select images and labels
    X, labels = X[ix], y[ix]
    # generate class labels
    target = np.ones((n_samples, 1))
    return [X, labels], target

# select real samples as suggested in the literature 
def generate_real_samples_smoothed(X, y, n_samples):
    # choose random instances
    ix = randint(0, X.shape[0], n_samples)
    # select images and labels
    X, labels = X[ix], y[ix]
    # generate class labels
    target = np.ones((n_samples, 1))*0.9 # change here if needed
    return [X, labels], target

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, args):

    n_classes = args.NUMBER_OF_CLASSES
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]
 

# generate points in latent space for saving progress
def generate_latent_points_fix(latent_dim, n_classes):

    # generate points in the latent space
    x_input = randn(latent_dim * n_classes)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_classes, latent_dim)
    # generate labels
    labels = np.array(list(range(n_classes)))
    return z_input, labels

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, args):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, args)
    # predict outputs
    #print(f"SAFE. z_input: {z_input.shape}, labels_input:{labels_input.shape}")
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y
    
# here for the model with only 3 classes
def save_progress_full(generator, z, label, epoch, class_header, args, prefix = ""):
    i=0
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,7))
    gen_imgs = generator.predict([z,label])
    
    for ax in axes:
        ax.imshow(gen_imgs[i] * 0.5 + 0.5)
        ax.set_title(f"{class_header[i]}")
        i=i+1
        ax.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, right=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Epoch: {epoch}")
    plt.savefig(os.path.join(args.FULL_SAVE, f"{prefix}facegen_epoch_{epoch}_full.png"), dpi=300)
    #plt.show()
    plt.close(fig=fig)
    
    
# here for the model with all 40 classes
def save_progress_full_40(generator, z, label, epoch, class_header, args, prefix = ""):
    i=0
    fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(20,20))
    gen_imgs = generator.predict([z,label])
    for row in axes:
        for ax in row:
            ax.imshow(gen_imgs[i] * 0.5 + 0.5)
            ax.set_title(f"{class_header[i]}")
            i=i+1
            ax.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, right=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Epoch: {epoch}")
    plt.savefig(os.path.join(args.FULL_SAVE, f"{prefix}facegen_epoch_{epoch}_full.png"), dpi=300)
    #plt.show()
    plt.close(fig=fig)
    
# independent of amount of classes
def save_progress_variety(generator, z, label, epoch, args, prefix = "", classname=""):
    i=0
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15,15))
    gen_imgs = generator.predict([z,label])
    for row in axes:
        for ax in row:
            ax.imshow(gen_imgs[i] * 0.5 + 0.5)
            i=i+1
            ax.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, right=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"{classname} on epoch: {epoch}")
    plt.savefig(os.path.join(args.CLASSES_SAVE, f"{classname}_{prefix}epoch_{epoch}.png"), dpi=300)
    plt.close(fig=fig)
    #plt.show()
    
"""
Weights MUST HAVE layout: "prefix_cgan_{generator, discriminator}_epoch_X.h5"

Actually DEPRECATED, because weights are loaded manually to ensure everything goes right
"""
def load_latest_model(d, g, path, prefix=""):
    files = os.listdir(path)
    print(files)
    #print(files)
    max_d = 0
    max_g = 0
    for file in files: 
        print(file.split("_")[0])
        if (file.split("_")[0] == prefix):
            #print(file.split("_")[2])
            print(file)
            if (file.split("_")[2] == "generator"):
                maxx = int(file.split("_")[::-1][0].split(".")[0])
                #print(maxx)
                if (maxx>max_g):
                    max_g = maxx
            if (file.split("_")[2] == "discriminator"):
                maxx = int(file.split("_")[::-1][0].split(".")[0])
                #print(maxx)
                if (maxx>max_d):
                    max_d = maxx

    g.load_weights(os.path.join(path,f"{prefix}_cgan_generator_epoch_{max_g}.h5"))
    d.load_weights(os.path.join(path,f"{prefix}_cgan_discriminator_epoch_{max_g}.h5"))
    
    print(f"Loaded weights for epoch: {max_g}")
    return max_g

def plot_random_with_discr(d_model, g_model, epoch, args,  latent_dim=100, save_pref=""):
    z, label = generate_latent_points_fix(latent_dim, n_classes=3)
    predd = g_model.predict([z, label])
    i=0
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,7))
    gen_imgs = g_model.predict([z,label])
    d_pred = d_model.predict([predd,label])
    
    for ax in axes:
        ax.imshow(gen_imgs[i] * 0.5 + 0.5)
        ax.set_title(f"{CLASS_HEADER[i]}: {np.round(d_pred[i][0]*100, decimals=2)}%")
        i=i+1

        ax.tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False, labelleft=False, labelright=False)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Epoch {epoch} - Descriminator probabilities of being a real image")
    if(save_pref):
        plt.savefig(os.path.join(args.DISCR_SAVE+f"{save_pref}discriminator_epoch_{epoch}.png"), dpi=300)
    #plt.show()
    plt.close()