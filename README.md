# Diagnosis
Hands on project to improve my programing skill
import kagglehub
pvlima_pretrained_pytorch_models_path = kagglehub.dataset_download('pvlima/pretrained-pytorch-models')
paultimothymooney_breast_histopathology_images_path = kagglehub.dataset_download('paultimothymooney/breast-histopathology-images')
allunia_breastcancermodel_path = kagglehub.dataset_download('allunia/breastcancermodel')

print('Data source import complete.')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


from glob import glob
from skimage.io import imread
from os import listdir

import time
import copy
from tqdm import tqdm_notebook as tqdm

run_training = False
retrain = False
find_learning_rate = False

Exploring the data structure
files = listdir("../input/breast-histopathology-images/")
print(len(files))

files[0:10]
files = listdir("../input/breast-histopathology-images/IDC_regular_ps50_idx5/")
len(files)
