import torch as T
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if T.cuda.is_available():
    device = T.device("cuda")

a = T.tensor([1, 2, 4]).to(device)