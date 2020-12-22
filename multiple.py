from os import listdir
from basic_model import model_training
from util import csv_to_dataset
import pandas as pd
import numpy as np
from models import simple_model

best_results = {}

history_points = 50

path = 'data_daily/'

files = [path + f for f in listdir(path)]

for file in files:

    best_results[file] = model_training(file, simple_model, history_points, offset=5)

print(best_results)



