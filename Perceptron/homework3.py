# %%
# Libraries 
import pandas as pd
import matplotlib.pyplot as plt

# Personal Libraries
from perceptron import Perceptron

# %%
data_root_path = './bank-note'
train_data = pd.read_csv('{0}/train.csv'.format(data_root_path), header=None)
test_data = pd.read_csv('{0}/test.csv'.format(data_root_path), header=None)
# %%
