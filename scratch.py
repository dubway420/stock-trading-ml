from util import csv_to_dataset


dataset = csv_to_dataset('data_daily/AAPL_daily.csv')

X = dataset[0]

Y = dataset[2]

