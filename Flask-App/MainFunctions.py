import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchmetrics import  PearsonCorrCoef
import torch.nn.functional as F
from random import randrange
import os, os.path
import pandas as pd
import numpy as np
from numpy import random
import cv2 as cv2
from PIL import Image, ImageOps
import torchmetrics
import warnings
warnings.filterwarnings("ignore")
import MainNetDefinitions as netdefs
import math
import copy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
device =  torch.device("cpu")
size = (256,256)
path = os.path.abspath(os.getcwd())

def trans_normalize(img):
    """
    Function for applying pixel normalization column wise to an image
    Args:
        img (np array): image to normalize pixels

    Returns:
        np array: normalized image
    """
    img = np.divide(img , img.sum(axis=0), out=np.zeros_like(img), where=img.sum(axis=0) > 0)
    return img

def trans_padding(img):
    """Function for resizing img using padding method as described in thesis

    Args:
        img (np array): array representing the img

    Returns:
        np array: img resized by padding method
    """
    width = 256 - img.shape[0]
    height = 256 - img.shape[1]
    if (height > 0) and (width > 0):
        img = np.pad(img,[(math.floor(width / 2),math.ceil(width / 2)),(math.floor(height / 2),math.ceil(height / 2))])
    else:
        # if padding is not possible because image is larger we resize it to 256
        img = cv2.resize(img,(size))
    return img.astype(np.float32)    


# code idea from https://stackoverflow.com/questions/46274961/removing-horizontal-lines-in-image-opencv-python-matplotlib
# morphological transf basis from https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def clearstraightlines(img):
    """ Function for removing dotted vertical lines using morphological transformations and otsus method

    Args:
        img (np array): array containing image depicting signal

    Returns:
        img (np array): img with removed dotted line
    """
    img = copy.copy(img)
    width = img.shape[1]
    height = img.shape[0]
   
    thresh = cv2.threshold(img, 0, 255,  cv2.THRESH_OTSU)[1]
    kernel = np.ones((10,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, math.floor(img.shape[0] * 0.8)))
    vertical_lines = cv2.morphologyEx(closing , cv2.MORPH_OPEN, vertical_kernel , iterations=2)

    img[np.nonzero(vertical_lines)] = 0

    #remove empty columns
    img = np.delete(img, np.argwhere(img.mean(axis=0) == 0), 1)
    img = cv2.resize(np.array(img),(width,height), cv2.INTER_NEAREST)    

    return img

transform_padding = transforms.Compose( [transforms.Lambda(clearstraightlines), transforms.Lambda(trans_padding), transforms.Lambda(trans_normalize), transforms.ToTensor()])

  
def buildmodelFromParams(netparams, load=False, seed=SEED):
    """Function for initializing model based on passed parameters

    Args:
        netparams (dict): dict with paramters to use for initialization
        load (bool, optional): flag controlling if saved model weights should be loaded. Defaults to False.
        seed (int, optional): random state. Defaults to SEED.

    Returns:
        Pytorch NN: initialized model instance
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    blocks = netparams["blocks"]
    kernel_size = netparams["kernel_size"]
    stride = netparams["stride"]
    padding = netparams["padding"]
    modelname = netparams["modelname"] if "modelname" in netparams else "dummy"
    torch.clear_autocast_cache(),  torch.manual_seed(seed)
    model = netdefs.HazelNet( blocks = blocks, kernel_size=kernel_size, padding=padding, stride=stride, seed=seed, modelname=modelname, transform=transform_padding).to(device, non_blocking=True)
    if load: 
        model.load_state_dict(torch.load(f"static/neuralnets/{modelname}", map_location=torch.device('cpu')))

    return model

def getparamcount(model):
    """Function for calculating the number of parameters in a model

    Args:
        model (Pytorch NN): neural network

    Returns:
        int: number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def importimg(path, invert):
    """ Function for importing an image. 
        Handling of transparency inverting and greyscaling of image

    Args:
        path (str): local path to uploaded image
        invert (bool): boolean regulationg image inversion if background is white

    Returns:
        np array: imported image
    """
    img = Image.open(path).convert("RGBA")
    if invert:
        background = Image.new('RGBA', img.size, (255,255,255))
        img = Image.alpha_composite(background, img).convert("L")
        img = ImageOps.invert(img)
    else: 
        background = Image.new('RGBA', img.size, (0,0,0)) 
        img = Image.alpha_composite(background, img).convert("L")
    return np.array(img)

