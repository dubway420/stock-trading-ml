import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
from os import listdir
from statistics import median as med
folder = "data_daily"

files = listdir(folder)

for file in files:

    data = pd.read_csv(folder + "/" + file)
    dates = data.get("date")[-500:]

    x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    open_price = data.get("1. open")[-500:]

    name = file.split("_")[0]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x, open_price, label=name)
    plt.gcf().autofmt_xdate()

every_nth = int(len(x)/9)

plt.xticks(x[::every_nth])

plt.legend()
plt.show()
