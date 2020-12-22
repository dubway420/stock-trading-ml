from basic_model import model_training
from models import simple_model


history_points = 50

path = 'data_daily/'

file = path + 'IBM_daily.csv'

for offset in range(5):

    best_results = model_training(file, simple_model, history_points, offset=offset)


